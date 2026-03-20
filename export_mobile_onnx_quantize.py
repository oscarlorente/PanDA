from __future__ import absolute_import, division, print_function
"""
ONNX Model Export Script for Mobile Deployment

"""
import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


from networks.models import *

class RandomCalibrationDataReader(CalibrationDataReader):
    """Generates synthetic random calibration data for static ONNX quantization.

    For best results, replace with real representative inputs using
    ImageCalibrationDataReader below.
    """
    def __init__(self, input_size, num_samples=100, input_name="input"):
        self.input_size = input_size
        self.num_samples = num_samples
        self.input_name = input_name
        self.current = 0

    def get_next(self):
        if self.current >= self.num_samples:
            return None
        self.current += 1
        # Simulate normalised image inputs in [0, 1]
        return {self.input_name: np.random.rand(*self.input_size).astype(np.float32)}

    def rewind(self):
        self.current = 0


class ImageCalibrationDataReader(CalibrationDataReader):
    """Loads real images from a directory as calibration data.

    Images are resized to model input dimensions and normalised to [0, 1].
    Pass --calibration-dir to use this reader instead of random data.
    """
    def __init__(self, image_dir, input_size, input_name="input"):
        import glob
        self.input_name = input_name
        self.input_size = input_size  # (batch, C, H, W)
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.images = [f for ext in exts for f in glob.glob(os.path.join(image_dir, ext))]
        if not self.images:
            raise FileNotFoundError(f"No .jpg/.jpeg/.png images found in {image_dir}")
        self.current = 0
        print(f"  Calibration: found {len(self.images)} images in {image_dir}")

    def get_next(self):
        if self.current >= len(self.images):
            return None
        from PIL import Image
        img = Image.open(self.images[self.current]).convert("RGB")
        img = img.resize((self.input_size[3], self.input_size[2]))  # PIL takes (W, H)
        data = np.array(img, dtype=np.float32) / 255.0
        data = data.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)
        self.current += 1
        return {self.input_name: data}

    def rewind(self):
        self.current = 0


class ModelWrapper(nn.Module):
    """Wrapper to extract only the depth prediction from the model output."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # Extract the depth prediction from the dict
        if isinstance(output, dict):
            return output['pred_depth']
        return output

def quantize_onnx_model(
    model_path,
    output_path,
    input_size,
    method='static',
    num_calibration_samples=100,
    calibration_dir=None,
):
    """
    Apply INT8 quantization to an exported FP32 ONNX model.

    For Android/XNNPACK deployment the only working method is **static** QDQ
    with uint8 activations (U8S8).  Dynamic quantization emits ConvInteger nodes
    (QOperator format) which are not implemented in the Android ORT XNNPACK EP.

    A pre-processing pass (shape inference + graph optimisation) is run
    automatically before quantization, as recommended by the ORT documentation.

    Args:
        model_path:               Path to the FP32 ONNX model.
        output_path:              Destination for the INT8 quantized model.
        input_size:               (batch, C, H, W) used for calibration.
        method:                   'static' (recommended, Android-compatible) or
                                  'dynamic' (desktop-only, not supported by
                                  Android XNNPACK — emits ConvInteger nodes).
        num_calibration_samples:  Random samples used when calibration_dir is None.
        calibration_dir:          Directory of real images for better calibration.
                                  Uses random data when None.
    """
    preprocessed_path = model_path.replace('.onnx', '_pre.onnx')

    print("\n[Quantization] Pre-processing: shape inference + graph optimisation...")
    try:
        quant_pre_process(model_path, preprocessed_path)
        print("  ✓ Pre-processing done")
    except Exception as e:
        print(f"  ⚠ Pre-processing failed ({e}), quantizing original model instead.")
        preprocessed_path = model_path

    try:
        if method == 'dynamic':
            # Dynamic quantization: weights quantized statically; activations at runtime.
            # ⚠ NOT suitable for Android deployment: produces ConvInteger (QOperator) nodes
            # which are not implemented in the Android ORT XNNPACK execution provider.
            # Use --quantize static for Android.
            print("[Quantization] ⚠  WARNING: dynamic quantization produces ConvInteger "
                  "nodes that are NOT supported by the Android ORT XNNPACK EP.")
            print("[Quantization]    Use --quantize static for Android/mobile deployment.")
            print("[Quantization] Applying dynamic INT8 quantization (desktop inference only)...")
            quantize_dynamic(
                preprocessed_path,
                output_path,
                weight_type=QuantType.QInt8,
            )
        else:  # static
            # Static QDQ with U8S8 (uint8 activations, int8 weights):
            # - XNNPACK EP on Android requires uint8 activations (QUInt8), not QInt8.
            # - S8S8 activations cause "not implemented" errors on Android ORT.
            if calibration_dir:
                reader = ImageCalibrationDataReader(
                    image_dir=calibration_dir,
                    input_size=input_size,
                    input_name="input",
                )
            else:
                print(f"[Quantization] No --calibration-dir provided; "
                      f"using {num_calibration_samples} random samples.\n"
                      "  Tip: pass --calibration-dir for higher post-quantization accuracy.")
                reader = RandomCalibrationDataReader(
                    input_size=input_size,
                    num_samples=num_calibration_samples,
                    input_name="input",
                )
            print("[Quantization] Applying static INT8 quantization (QDQ, U8S8 — Android/XNNPACK compatible)...")
            quantize_static(
                preprocessed_path,
                output_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,  # uint8 required by Android XNNPACK EP
                weight_type=QuantType.QInt8,
            )

        print(f"  ✓ Quantized model saved to {output_path}")
    finally:
        if preprocessed_path != model_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


def export_model_with_onnx(config_path, output_path, example_input_size=(1, 3, 504, 1008), backend='onnx'):
    """
    Export model using ONNX for mobile deployment.
    
    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.onnx)
        example_input_size: Tuple of (batch, channels, height, width) for export
        backend: Backend to use
    """
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Load model
    print("Loading model...")
    model_dict = torch.load(config["model_path"], map_location='cpu')
    
    # Create model
    model = make(config['model'])
    
    # Remove DataParallel wrapper if present
    if any(key.startswith('module') for key in model_dict.keys()):
        # Remove 'module.' prefix from keys
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
    
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()
    
    # Wrap model to extract only depth output
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    print(f"Model loaded successfully. Setting to evaluation mode...")
    
    # Create example input for export
    sample_inputs = (torch.randn(example_input_size),)
    
    # Define backend_name early for error handling
    backend_name = backend.lower()
    
    print(f"Exporting model with input size {example_input_size}...")
    try:
        # Export the model using torch.onnx
        print("Running torch.onnx.export()...")

        if backend_name == 'onnx':
            torch.onnx.export(
                wrapped_model,
                sample_inputs,
                output_path,
                input_names=["input"],
                output_names=["depth"],
                opset_version=18,          # 18+ required; Resize op has no downgrade adapter to 17
                dynamic_axes=None,         # fixed shape — matches your CoreML approach
            )

        print(f"✓ Successfully exported model to {output_path}")
        print(f"  Model input size: {example_input_size}")
        print(f"  Backend: {backend_name.upper()}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error during model export: {error_msg}")
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
    parser.add_argument('--backend', type=str, default='onnx',
                        choices=['onnx'],
                        help='Backend for optimization: onnx (ONNX FP32)')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'static', 'dynamic'],
                        help='Post-export INT8 quantization method. '
                             'static: QDQ U8S8, recommended for CNNs; required for Android/XNNPACK. '
                             'dynamic: no calibration needed but produces ConvInteger nodes '
                             'NOT supported by Android ORT XNNPACK EP — desktop only. '
                             'Default: none')
    parser.add_argument('--calibration-dir', type=str, default=None,
                        help='Directory of representative .jpg/.png images for static '
                             'quantization calibration. '
                             'Falls back to random data when not provided.')
    parser.add_argument('--calibration-samples', type=int, default=100,
                        help='Number of random calibration samples when --calibration-dir '
                             'is not supplied (default: 100)')

    args = parser.parse_args()
    
    # Calculate input dimensions
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)
    
    print("ONNX Model Export for Mobile")
    success = export_model_with_onnx(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        backend=args.backend,
    )
    
    if success:
        print("Export completed successfully!")

    if success and args.quantize != 'none':
        base, ext = os.path.splitext(args.output)
        quantized_output = f"{base}_int8_{args.quantize}{ext}"
        quantize_onnx_model(
            model_path=args.output,
            output_path=quantized_output,
            input_size=input_size,
            method=args.quantize,
            num_calibration_samples=args.calibration_samples,
            calibration_dir=args.calibration_dir,
        )
        print(f"\nQuantization completed! INT8 model: {quantized_output}")
        