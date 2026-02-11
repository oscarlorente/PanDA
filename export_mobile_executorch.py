from __future__ import absolute_import, division, print_function
"""
ExecuTorch Model Export Script for Mobile Deployment

Backend Selection Guide:
- XNNPACK (Recommended): Best for CPU-based mobile inference
  * Excellent operator coverage
  * Optimized for ARM/x86 processors
  * Supports complex operations (slice_scatter, indexing, etc.)
  * Supports 8-bit quantization for smaller models and faster inference

- CoreML: Apple Neural Engine acceleration (iOS/macOS)
  * Optimized for Apple devices (iPhone, iPad, Mac)
  * Hardware-accelerated inference using ANE
  * Best performance on Apple Silicon
  * Requires macOS for export

- Vulkan: GPU-accelerated inference (Experimental)
  * Limited operator support (may skip operations)
  * Best for simple models or when GPU is critical
  * Some ops fall back to CPU: slice_scatter, copy, index, floor

- CPU: Basic fallback
  * Supports all operators
  * No optimizations
  * Use for testing/compatibility

If Vulkan export fails, try XNNPACK instead.
"""
import argparse
import yaml
import torch
import torch.nn as nn

from executorch.exir import to_edge_transform_and_lower

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

def export_model_with_executorch(config_path, output_path, example_input_size=(1, 3, 504, 1008), backend='vulkan'):
    """
    Export model using ExecuTorch for mobile deployment.
    
    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.pte)
        example_input_size: Tuple of (batch, channels, height, width) for export
        backend: Backend to use ('vulkan' for GPU, 'xnnpack' for CPU, 'cpu' for basic CPU)
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
        # Export the model using torch.export
        print("Running torch.export.export()...")
        
        exported_program = torch.export.export(wrapped_model, sample_inputs)
        
        print("Converting to ExecuTorch format...")
        # Lower to edge dialect and apply partitioners based on backend
        partitioner = []
        
        if backend_name == 'vulkan':
            print("Using Vulkan partitioner for GPU acceleration...")
            
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
            partitioner = [VulkanPartitioner()]
        
        elif backend_name == 'xnnpack':
            print("Using XNNPACK partitioner for CPU optimization...")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            partitioner = [XnnpackPartitioner()]
        
        elif backend_name == 'coreml':
            print("Using CoreML partitioner for Apple Neural Engine acceleration...")
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
            partitioner = [CoreMLPartitioner()]
        
        else:
            print("Using basic CPU backend (no partitioner)...")
            partitioner = []
        
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioner,
        ).to_executorch()

        # Save the ExecuTorch program
        print(f"Saving ExecuTorch model to {output_path}...")
        with open(output_path, "wb") as file:
            file.write(executorch_program.buffer)
        
        print(f"✓ Successfully exported ExecuTorch model to {output_path}")
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
    parser = argparse.ArgumentParser(description='Export model using ExecuTorch for mobile deployment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='./checkpoints/model_mobile.pte',
                        help='Output path for ExecuTorch model (.pte)')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width will be 2x height for ERP). Must be a multiple of 14.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for example input')
    parser.add_argument('--backend', type=str, default='vulkan',
                        choices=['vulkan', 'vulkan-compat', 'xnnpack', 'coreml', 'cpu'],
                        help='Backend for optimization: vulkan (GPU), vulkan-compat (GPU with workarounds), xnnpack (CPU optimized), coreml (Apple Neural Engine), cpu (basic)')

    args = parser.parse_args()
    
    # Calculate input dimensions
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)
    
    print("ExecuTorch Model Export for Mobile")
    success = export_model_with_executorch(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        backend=args.backend,
    )
    
    if success:
        print("Export completed successfully!")
        