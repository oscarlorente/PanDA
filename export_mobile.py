from __future__ import absolute_import, division, print_function
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from networks.models import *


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


def export_model_for_mobile(config_path, output_path, example_input_size=(1, 3, 504, 1008), backend='CPU'):
    """
    Export and optimize model for mobile deployment.
    
    Args:
        config_path: Path to config YAML file
        output_path: Path where optimized model will be saved (.pt or .ptl)
        example_input_size: Tuple of (batch, channels, height, width) for tracing
        backend: Backend to optimize for ('CPU' or 'vulkan' for Android GPU)
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
    
    # Create example input for tracing
    example_input = torch.randn(example_input_size)
    
    print(f"Tracing model with input size {example_input_size}...")
    try:
        # Trace the model (strict=False to allow dict outputs and dynamic shapes)
        traced_model = torch.jit.trace(wrapped_model, example_input, strict=False)
        
        print(f"Optimizing for mobile (backend: {backend})...")
        # Apply mobile optimizations (operator fusion, weight pre-packing)
        optimized_model = optimize_for_mobile(traced_model, backend=backend)
        
        # Save model using lite interpreter for smaller size
        print(f"Saving optimized model to {output_path}...")
        optimized_model._save_for_lite_interpreter(output_path)
        
        print(f"✓ Successfully exported mobile-optimized model to {output_path}")
        print(f"  Model input size: {example_input_size}")
        print(f"  Backend: {backend}")
        
        # Verify the saved model
        if backend.lower() == 'vulkan':
            print("\nNote: Vulkan-optimized model exported successfully.")
            print("  Vulkan verification requires a device with Vulkan support.")
            print("  Skipping CPU-based verification for vulkan model.")
            print("  Please test on your target Android device.")
        else:
            print("\nVerifying saved model on CPU...")
            loaded_model = torch.jit.load(output_path, map_location='cpu')
            test_output = loaded_model(example_input)
            print(f"✓ Model verification successful. Output shape: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during model export: {str(e)}")
        print("\nTrying alternative export method (standard TorchScript)...")
        
        try:
            # Fallback: save as standard TorchScript
            traced_model = torch.jit.trace(wrapped_model, example_input, strict=False)
            fallback_path = output_path.replace('.ptl', '.pt')
            torch.jit.save(traced_model, fallback_path)
            print(f"✓ Saved as standard TorchScript to {fallback_path}")
            return True
        except Exception as e2:
            print(f"✗ Fallback export also failed: {str(e2)}")
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export and optimize model for mobile deployment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='./checkpoints/model_mobile.ptl',
                        help='Output path for optimized model (.ptl for lite interpreter, .pt for standard)')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width will be 2x height for ERP). Must be a multiple of 14.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for example input')
    parser.add_argument('--backend', type=str, default='CPU',
                        choices=['CPU', 'vulkan'],
                        help='Backend for mobile optimization (CPU or vulkan for Android GPU)')

    args = parser.parse_args()
    
    # Calculate input dimensions
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)
    
    print("=" * 70)
    print("PyTorch Mobile Model Export")
    print("=" * 70)
    
    success = export_model_for_mobile(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        backend=args.backend,
    )
    
    if success:
        print("\n" + "=" * 70)
        print("Export completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. For iOS: Use the .ptl file in your iOS app with PyTorch Mobile")
        print("2. For Android: Use the .ptl file in your Android app with PyTorch Mobile")
        print("3. Performance tips:")
        print("   - Reuse input tensors in your Android app (avoid allocation per inference)")
        print("   - Use org.pytorch.Tensor.allocateFloatBuffer() and reuse the buffer")
        print("4. Integration guide: https://pytorch.org/mobile/")
    else:
        print("\n" + "=" * 70)
        print("Export failed. Please check the error messages above.")
        print("=" * 70)
