from __future__ import absolute_import, division, print_function
"""
ONNX Model Export Script for Mobile Deployment

"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.onnx


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

def export_model_with_onnx(config_path, output_path, example_input_size=(1, 3, 504, 1008)):
    """
    Export model using ONNX for mobile deployment.
    
    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.onnx)
        example_input_size: Tuple of (batch, channels, height, width) for export
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
    
    print(f"Exporting model with input size {example_input_size}...")
    try:
        # Export the model using torch.onnx
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

        print(f"✓ Successfully exported model to {output_path}")
        print(f"  Model input size: {example_input_size}")
        
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
    )
    
    if success:
        print("Export completed successfully!")
        