import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        """
        DPTHead is always built with 4 encoder layers for checkpoint compatibility.
        The number of encoder layers used at inference time is controlled by
        DepthAnythingV2.num_encoder_layers and passed via layer_indices_used.
        """
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                for _ in range(4)
            ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, out_features, patch_h, patch_w, layer_indices_used):
        """
        Args:
            out_features:       encoder feature tuples (patch_tokens, cls_token)
            patch_h, patch_w:   spatial patch grid dimensions
            layer_indices_used: encoder layer indices that were extracted,
                                used to select the correct project/resize heads.
        """
        # Map each extracted layer index to its decoder head position.
        # This lookup table covers all supported encoders — the head position
        # is the 0-based rank of the layer index within the full 4-layer list.
        full_layer_lists = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }
        # Determine which full list applies by inspecting the indices given
        # We match by checking which full list contains all of layer_indices_used
        head_indices = None
        for full_list in full_layer_lists.values():
            if all(i in full_list for i in layer_indices_used):
                head_indices = [full_list.index(i) for i in layer_indices_used]
                break
        if head_indices is None:
            # Fallback: assume positional order
            head_indices = list(range(len(layer_indices_used)))

        out = []
        for x, head_i in zip(out_features, head_indices):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[head_i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[head_i](x)
            x = self.resize_layers[head_i](x)
            out.append(x)

        if len(out) == 4:
            layer_1, layer_2, layer_3, layer_4 = out

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)

        elif len(out) == 2:
            # 2-layer fast path: reuse the coarse decoder heads from checkpoint
            layer_3, layer_4 = out
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            out = self.scratch.output_conv1(path_3)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)

        else:
            raise ValueError(f"Expected 2 or 4 encoder features, got {len(out)}")

        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
        num_encoder_layers=4,
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            'vits': {4: [2, 5, 8, 11],   2: [8, 11]},
            'vitb': {4: [2, 5, 8, 11],   2: [8, 11]},
            'vitl': {4: [4, 11, 17, 23], 2: [17, 23]},
            'vitg': {4: [9, 19, 29, 39], 2: [29, 39]},
        }

        self.max_depth = max_depth
        self.encoder = encoder
        self.num_encoder_layers = num_encoder_layers
        self.pretrained = DINOv2(model_name=encoder)

        # Always build with 4 layers for checkpoint compatibility
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    def forward(self, x):
        t0 = time.time()

        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        layer_indices = self.intermediate_layer_idx[self.encoder][self.num_encoder_layers]
        features = self.pretrained.get_intermediate_layers(
            x, layer_indices, return_class_token=True, norm=True
        )

        t1 = time.time()

        # Opt 3: DPTHead output_conv2 ends with Sigmoid → shape [B, 1, H, W].
        # We multiply by max_depth and return [B, H, W] (squeeze dim 1).
        # panda.py then does unsqueeze(1) to get [B, 1, H, W] back — that
        # squeeze+unsqueeze roundtrip is eliminated here by NOT squeezing,
        # so panda.py must also be updated to not unsqueeze.
        depth = self.depth_head(features, patch_h, patch_w, layer_indices) * self.max_depth
        # depth shape: [B, 1, H, W]

        t2 = time.time()
        print(f"[DepthAnythingV2] Encoder: {t1-t0:.3f}s  Decoder: {t2-t1:.3f}s  "
              f"(num_encoder_layers={self.num_encoder_layers})")

        # Return [B, H, W] — consistent with original API
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        depth = self.forward(image)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
