import os
import cv2
import time
import torch
import yaml
import numpy as np
import torch.nn as nn
from executorch.runtime import Runtime
from networks.models import *
from torchvision.transforms import Compose
from depth_anything_utils import Resize, NormalizeImage, PrepareForNet
import matplotlib

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        output = self.model(x)
        return output['pred_depth'] if isinstance(output, dict) else output

if __name__ == '__main__':
    erp_height = 504
    filename = './calib_images/equirectangular_2544150_6987219e1638d.jpeg'
    out_dir = './executorch_test_output'
    os.makedirs(out_dir, exist_ok=True)
    DEVICE = 'cpu'

    # transform
    erp_width = 2 * erp_height
    transform = Compose([
        Resize(
            width=erp_width,
            height=erp_height,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    raw_image = cv2.imread(filename)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    image_torch = torch.from_numpy(transform({'image': image})['image']).unsqueeze(0).to(DEVICE)

    for pte_filename in [
            "panda_model_mobile_504_xnnpack",
            "panda_model_mobile_504_xnnpack_merged_lora_no_fuse_layerscale_4_layers",
            "panda_model_mobile_504_xnnpack_merged_lora_fuse_layerscale_4_layers",
            "panda_model_mobile_504_xnnpack_merged_lora_4_layers",
            "panda_model_mobile_504_xnnpack_merged_lora_2_layers",
        ]:
        rt = Runtime.get()
        prog = rt.load_program(f"./checkpoints/{pte_filename}.pte")
        m = prog.load_method("forward")
        t = time.time()
        depth = m.execute([image_torch])[0]
        print(f"Execution time for {pte_filename}: {time.time() - t} seconds")
        depth = depth[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + f'_{pte_filename}.png')
        cv2.imwrite(output_path, depth)

    exit()

    # # ── Load model ──────────────────────────────────────────────────────────────
    # with open("./config/inference/custom_vits.yaml", 'r') as f:
    #     config = yaml.load(f, yaml.FullLoader)

    # model_dict = torch.load(config["model_path"], map_location='cpu')
    # if any(k.startswith('module') for k in model_dict.keys()):
    #     model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

    # model = make(config['model'])
    # model.load_state_dict({k: v for k, v in model_dict.items() if k in model.state_dict()})
    # model.eval()
    # wrapped = ModelWrapper(model).eval()

    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # test_input = (torch.rand(1, 3, 504, 1008) - mean) / std

    # # ── Test 1: FP32 eager (sanity check) ───────────────────────────────────────
    # with torch.no_grad():
    #     out_fp32 = wrapped(test_input)
    # print("=== FP32 eager ===")
    # print("min:", out_fp32.min().item(), "max:", out_fp32.max().item(), "std:", out_fp32.std().item())

    # # ── Test 2: FP16 eager ──────────────────────────────────────────────────────
    # wrapped_fp16 = ModelWrapper(model).eval().half()
    # with torch.no_grad():
    #     out_fp16 = wrapped_fp16(test_input.half())
    # print("\n=== FP16 eager ===")
    # print("min:", out_fp16.min().item(), "max:", out_fp16.max().item(), "std:", out_fp16.std().item())


    # ── Test 3: ExecuTorch FP32 xnnpack .pte ─────────────────────────────────────
    try:
        rt = Runtime.get()
        prog = rt.load_program("./checkpoints/panda_model_mobile_504_xnnpack.pte")
        m = prog.load_method("forward")
        t0 = time.time()
        out = m.execute([test_input])[0]
        t1 = time.time()
        print("\n=== ExecuTorch FP32 xnnpack ===")
        print("min:", out.min().item(), "max:", out.max().item(), "std:", out.std().item())
        print("Execution time:", t1 - t0, "seconds")
    except Exception as e:
        print("\n=== ExecuTorch FP32 xnnpack FAILED ===", e)

    try:
        rt = Runtime.get()
        prog = rt.load_program("./checkpoints/panda_model_mobile_504_xnnpack_no_merge_lora.pte")
        m = prog.load_method("forward")
        t0 = time.time()
        out = m.execute([test_input])[0]
        t1 = time.time()
        print("\n=== ExecuTorch FP32 xnnpack (no merge LoRA) ===")
        print("min:", out.min().item(), "max:", out.max().item(), "std:", out.std().item())
        print("Execution time:", t1 - t0, "seconds")
    except Exception as e:
        print("\n=== ExecuTorch FP32 xnnpack (no merge LoRA) FAILED ===", e)

    try:
        rt = Runtime.get()
        prog = rt.load_program("./checkpoints/panda_model_mobile_504_xnnpack_merged_lora_4_layers.pte")
        m = prog.load_method("forward")
        t0 = time.time()
        out = m.execute([test_input])[0]
        t1 = time.time()
        print("\n=== ExecuTorch FP32 xnnpack (merged LoRA 4 layers) ===")
        print("min:", out.min().item(), "max:", out.max().item(), "std:", out.std().item())
        print("Execution time:", t1 - t0, "seconds")
    except Exception as e:
        print("\n=== ExecuTorch FP32 xnnpack (merged LoRA 4 layers) FAILED ===", e)

    try:
        rt = Runtime.get()
        prog = rt.load_program("./checkpoints/panda_model_mobile_504_xnnpack_merged_lora_2_layers.pte")
        m = prog.load_method("forward")
        t0 = time.time()
        out = m.execute([test_input])[0]
        t1 = time.time()
        print("\n=== ExecuTorch FP32 xnnpack (merged LoRA 2 layers) ===")
        print("min:", out.min().item(), "max:", out.max().item(), "std:", out.std().item())
        print("Execution time:", t1 - t0, "seconds")
    except Exception as e:
        print("\n=== ExecuTorch FP32 xnnpack (merged LoRA 2 layers) FAILED ===", e)

    # # ── Test 4: ExecuTorch FP16 xnnpack - with diagnostic ─────────────────────
    # try:
    #     prog_fp16 = rt.load_program("./checkpoints/panda_model_mobile_504_xnnpack_fp16.pte")
    #     m_fp16 = prog_fp16.load_method("forward")

    #     # The model expects FP32 input (wrapper casts internally)
    #     out_fp16_et = m_fp16.execute([test_input])[0]
    #     print("\n=== ExecuTorch FP16 xnnpack (FP32 input via wrapper) ===")
    #     print("dtype:", out_fp16_et.dtype)
    #     print("min:", out_fp16_et.min().item(), "max:", out_fp16_et.max().item(), "std:", out_fp16_et.std().item())

    #     # Also try with a constant input to see if output varies at all
    #     zero_input = torch.zeros(1, 3, 504, 1008)
    #     out_zero = m_fp16.execute([zero_input])[0]
    #     print("Output for zero input - min:", out_zero.min().item(), "max:", out_zero.max().item())

    #     ones_input = torch.ones(1, 3, 504, 1008)
    #     out_ones = m_fp16.execute([ones_input])[0]
    #     print("Output for ones input - min:", out_ones.min().item(), "max:", out_ones.max().item())

    # except Exception as e:
    #     print("\n=== ExecuTorch FP16 xnnpack FAILED ===", e)