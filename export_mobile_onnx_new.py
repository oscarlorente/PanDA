from __future__ import absolute_import, division, print_function
"""
ONNX Model Export Script for Mobile Deployment

Optimization flags:
  --merge-lora        Merge LoRA A/B matrices into base QKV weights (default on).
                      Zero accuracy change, eliminates LoRA runtime overhead.

  --fuse-layer-scale  Fuse LayerScale gamma into attn.proj and mlp.fc2 weights
                      (default on). Zero accuracy change, eliminates 24 element-
                      wise multiplications per forward pass (2 per ViT block × 12).

  --encoder-layers 2  Use only 2 intermediate encoder layers instead of 4.
                      Model always loads from the full 4-layer checkpoint.
                      Expect ~20-30% encoder speedup with some quality loss.
                      Evaluate quality before using in production.

  --fp16              Export with FP16 weights. Uses FP32 I/O with internal FP16
                      so no runtime code changes are needed.
"""
import argparse
import copy
import yaml
import torch
import torch.nn as nn
import torch.onnx


from networks.models import *


class ModelWrapperFP16(nn.Module):
    """FP32 I/O with internal FP16 — no runtime code changes needed."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.half()
        output = self.model(x)
        if isinstance(output, dict):
            return output['pred_depth'].float()
        return output.float()


class ModelWrapperFP32(nn.Module):
    """Wrapper to extract only the depth prediction from the model output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output['pred_depth']
        return output


def export_model_with_onnx(
    config_path,
    output_path,
    example_input_size=(1, 3, 504, 1008),
    use_fp16=False,
    merge_lora=True,
    fuse_layer_scale=True,
    num_encoder_layers=4,
):
    """
    Export model using ONNX for mobile deployment.

    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.onnx)
        example_input_size: Tuple of (batch, channels, height, width) for export
        use_fp16: FP32 I/O with internal FP16 weights
        merge_lora: Merge LoRA A/B matrices into base QKV weights before export
        fuse_layer_scale: Fuse LayerScale gamma into linear weights before export
        num_encoder_layers: Number of encoder layers at inference (2 or 4)
    """
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading model...")
    model_dict = torch.load(config["model_path"], map_location='cpu')

    # Always load with num_encoder_layers=4 for checkpoint weight compatibility.
    # We switch to the requested inference mode after loading.
    model_config = config['model']
    if num_encoder_layers != 4:
        model_config = copy.deepcopy(model_config)
        model_config['args']['num_encoder_layers'] = 4   # force 4-layer build

    model = make(model_config)

    if any(key.startswith('module') for key in model_dict.keys()):
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    # Switch to 2-layer inference mode AFTER loading weights
    if num_encoder_layers != 4:
        print(f"Switching to {num_encoder_layers}-layer inference mode...")
        model.core.num_encoder_layers = num_encoder_layers

    # --- Opt 6 (LoRA merge) ---
    if merge_lora:
        try:
            from networks.utils import merge_lora_weights, _LoRA_qkv
            if hasattr(model, 'core') and hasattr(model.core, 'pretrained'):
                merge_lora_weights(model.core)
            else:
                print("  Warning: could not find model.core.pretrained — skipping LoRA merge.")
        except ImportError as e:
            print(f"  Warning: could not import merge_lora_weights ({e}) — skipping.")
    else:
        print("LoRA merge skipped.")

    # --- Opt 4 (LayerScale fusion) ---
    if fuse_layer_scale:
        try:
            from depth_anything_v2_metric.depth_anything_v2.dinov2_layers.layer_scale import fuse_layer_scale_into_linear
            if hasattr(model, 'core') and hasattr(model.core, 'pretrained'):
                fuse_layer_scale_into_linear(model.core.pretrained)
            else:
                print("  Warning: could not find model.core.pretrained — skipping LayerScale fusion.")
        except ImportError as e:
            print(f"  Warning: could not import fuse_layer_scale_into_linear ({e}) — skipping.")
    else:
        print("LayerScale fusion skipped.")

    # Choose wrapper
    if use_fp16:
        print("FP16 mode: FP32 I/O with internal FP16.")
        wrapped_model = ModelWrapperFP16(model).eval()
        wrapped_model = wrapped_model.half()
        sample_inputs = (torch.randn(example_input_size),)
    else:
        print("FP32 mode.")
        wrapped_model = ModelWrapperFP32(model).eval()
        sample_inputs = (torch.randn(example_input_size),)

    precision_label = "FP16 (FP32 I/O)" if use_fp16 else "FP32"
    print(f"  Precision         : {precision_label}")
    print(f"  Encoder layers    : {num_encoder_layers}")
    print(f"  LoRA merged       : {merge_lora}")
    print(f"  LayerScale fused  : {fuse_layer_scale}")
    print(f"  Input size        : {example_input_size}")

    print(f"Exporting model with input size {example_input_size}...")
    try:
        print("Running torch.onnx.export()...")

        torch.onnx.export(
            wrapped_model,
            sample_inputs,
            output_path,
            input_names=["input"],
            output_names=["depth"],
            opset_version=18,          # 18+ required; Resize op has no downgrade adapter to 17
            dynamic_axes=None,         # fixed shape — matches your CoreML approach
        )

        print(f"✓ Export complete: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model using ONNX for mobile deployment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='./checkpoints/panda_model_mobile_504.onnx',
                        help='Output path for ONNX model (.onnx)')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width will be 2x height for ERP). Must be a multiple of 14.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for example input')
    parser.add_argument('--fp16', action='store_true',
                        help='FP32 I/O with internal FP16.')
    parser.add_argument('--no-merge-lora', action='store_true',
                        help='Skip LoRA weight merging.')
    parser.add_argument('--no-fuse-layer-scale', action='store_true',
                        help='Skip LayerScale fusion into linear weights.')
    parser.add_argument('--encoder-layers', type=int, default=4, choices=[2, 4],
                        help='Encoder layers at inference (2 or 4). Default: 4.')

    args = parser.parse_args()

    input_size = (args.batch_size, 3, args.height, args.height * 2)

    print("ONNX Model Export for Mobile")
    success = export_model_with_onnx(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        use_fp16=args.fp16,
        merge_lora=not args.no_merge_lora,
        fuse_layer_scale=not args.no_fuse_layer_scale,
        num_encoder_layers=args.encoder_layers,
    )

    if success:
        print("Export completed successfully!")
