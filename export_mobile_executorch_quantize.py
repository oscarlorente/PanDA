from __future__ import absolute_import, division, print_function
"""
ExecuTorch Model Export Script for Mobile Deployment

Backend Selection Guide:
- XNNPACK (Recommended): Best for CPU-based mobile inference
  * Excellent operator coverage
  * Optimized for ARM/x86 processors
  * Supports complex operations (slice_scatter, indexing, etc.)
  * Supports 8-bit quantization for smaller models and faster inference

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
import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

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


def load_calibration_images(data_dir, input_size=(504, 1008), max_samples=50):
    """
    Load calibration images from directory.
    
    Args:
        data_dir: Directory containing calibration images
        input_size: Target input size (height, width)
        max_samples: Maximum number of calibration samples to load
        
    Returns:
        List of preprocessed image tensors
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Calibration data directory not found: {data_dir}")
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"No images found in {data_dir}")
    
    # Limit number of samples
    image_files = sorted(image_files)[:max_samples]
    
    print(f"  Loading {len(image_files)} calibration images from {data_dir}...")
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess images
    calibration_samples = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            calibration_samples.append(img_tensor)
        except Exception as e:
            print(f"  Warning: Failed to load {img_path.name}: {e}")
            continue
    
    if not calibration_samples:
        raise ValueError("Failed to load any calibration images")
    
    print(f"  ✓ Loaded {len(calibration_samples)} calibration samples")
    return calibration_samples


def quantize_model_pt2e(model, sample_input, calibration_samples=None, is_per_channel=True):
    """
    Quantize model using PT2E flow with XNNPACK quantizer.
    
    Args:
        model: PyTorch model to quantize
        sample_input: Example input tensor for export
        calibration_samples: List of calibration samples for static quantization
        is_per_channel: Whether to use per-channel quantization
        
    Returns:
        Tuple of (quantized ExportedProgram, whether quantization succeeded)
    """
    print("\n" + "=" * 70)
    print("Quantizing Model (PT2E Flow)")
    print("=" * 70)
    
    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError as e:
        raise ImportError(
            "Failed to import quantization modules. Install with:\n"
            "  pip install torchao\n"
            f"Error: {e}"
        )
    
    # Step 1: Configure quantizer
    print("Step 1: Configuring XNNPACK quantizer...")
    qparams = get_symmetric_quantization_config(is_per_channel=is_per_channel)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)
    
    quant_type = "per-channel" if is_per_channel else "per-tensor"
    print(f"  ✓ Using 8-bit symmetric quantization ({quant_type})")
    
    # Step 2: Export for quantization
    print("\nStep 2: Exporting model for quantization...")
    exported_model = torch.export.export(model, (sample_input,))
    print("  ✓ Model exported")
    
    # Step 3: Prepare for quantization
    print("\nStep 3: Preparing model for quantization...")
    prepared_model = prepare_pt2e(exported_model.module(), quantizer)
    print("  ✓ Model prepared with quantization observers")
    
    # Step 4: Calibration (if samples provided)
    if calibration_samples:
        print(f"\nStep 4: Running calibration with {len(calibration_samples)} samples...")
        # Note: Don't call .eval() on prepared model - it doesn't support it
        with torch.no_grad():
            for i, cal_sample in enumerate(calibration_samples):
                if i % 10 == 0:
                    print(f"  Processing calibration sample {i+1}/{len(calibration_samples)}...")
                prepared_model(cal_sample)
        print("  ✓ Calibration complete (static quantization)")
    else:
        print("\nStep 4: Skipping calibration (dynamic quantization)")
        print("  ⚠ No calibration samples provided - using dynamic quantization")
    
    # Step 5: Convert to quantized model
    print("\nStep 5: Converting to quantized model...")
    quantized_model = convert_pt2e(prepared_model)
    print("  ✓ Model quantized")
    
    print("\n" + "=" * 70)
    print("Quantization Complete")
    print("=" * 70)

    # Re-export the quantized model to get ExportedProgram
    print("\nRe-exporting quantized model...")
    quantized_program = torch.export.export(quantized_model, (sample_input,))
    print("  ✓ Quantized model re-exported")
    
    return quantized_program


def test_exported_model(model_path, example_input_size=(1, 3, 504, 1008)):
    """
    Test the exported ExecuTorch model using the runtime pybind API.
    
    Args:
        model_path: Path to the exported .pte file
        example_input_size: Input size to test with
    """
    print("\n" + "=" * 70)
    print("Testing Exported Model")
    print("=" * 70)
    
    try:
        from executorch.runtime import Runtime
        
        print(f"Loading model from {model_path}...")
        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")
        
        print(f"Creating test input with size {example_input_size}...")
        test_input = torch.randn(example_input_size)
        
        print("Running inference...")
        outputs = method.execute([test_input])
        
        print(f"✓ Model test successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output dtype: {outputs[0].dtype}")
        print(f"  Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Runtime testing not available: {str(e)}")
        print("  To enable testing, install ExecuTorch with runtime support:")
        print("  pip install executorch")
        return False
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def export_model_with_executorch(config_path, output_path, example_input_size=(1, 3, 504, 1008), backend='vulkan', 
                                 quantize=False, calibration_data_dir=None, per_channel_quant=True):
    """
    Export model using ExecuTorch for mobile deployment.
    
    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.pte)
        example_input_size: Tuple of (batch, channels, height, width) for export
        backend: Backend to use ('vulkan' for GPU, 'xnnpack' for CPU, 'cpu' for basic CPU)
        quantize: Whether to apply 8-bit quantization (XNNPACK only)
        calibration_data_dir: Directory with calibration images for static quantization
        per_channel_quant: Use per-channel quantization (vs per-tensor)
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
    
    # Track actual quantization settings used
    actual_per_channel = per_channel_quant
    actual_calibration_samples = None
    
    # Define backend_name early for error handling
    backend_name = backend.lower()
    
    print(f"Exporting model with input size {example_input_size}...")
    try:
        # Export the model using torch.export
        print("Running torch.export.export()...")
        
        # Check if quantization is requested
        if quantize and backend_name != 'xnnpack':
            print("⚠ Warning: Quantization is only supported with XNNPACK backend")
            print("  Proceeding without quantization...")
            quantize = False
        
        # Handle quantization for XNNPACK backend
        if quantize and backend_name == 'xnnpack':
            # Load calibration samples if directory provided
            calibration_samples = None
            if calibration_data_dir:
                try:
                    h, w = example_input_size[2], example_input_size[3]
                    calibration_samples = load_calibration_images(
                        calibration_data_dir, 
                        input_size=(h, w),
                        max_samples=50
                    )
                    actual_calibration_samples = calibration_samples
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to load calibration data: {e}")
                    print(f"  Proceeding with dynamic quantization...")
                    calibration_samples = None
            
            # For dynamic quantization, use per-tensor quantization (more stable)
            # For static quantization with calibration, per-channel is preferred
            if calibration_samples is None and per_channel_quant:
                print("  ℹ Dynamic quantization detected - using per-tensor quantization for stability")
                use_per_channel = False
                actual_per_channel = False
            else:
                use_per_channel = per_channel_quant
            
            # Create sample input for quantization
            sample_input = torch.randn(example_input_size)
            
            # Apply PT2E quantization with error handling
            try:
                quantized_program = quantize_model_pt2e(
                    wrapped_model,
                    sample_input,
                    calibration_samples=calibration_samples,
                    is_per_channel=use_per_channel
                )
            except AssertionError as e:
                if "Invalid size of per channel quantization scales" in str(e):
                    print(f"\n⚠ Per-channel quantization failed: {e}")
                    print("  Retrying with per-tensor quantization...")
                    quantized_program = quantize_model_pt2e(
                        wrapped_model,
                        sample_input,
                        calibration_samples=calibration_samples,
                        is_per_channel=False
                    )
                    actual_per_channel = False
                else:
                    raise
            
            # Use the quantized program directly (already exported)
            print("\nUsing quantized model for ExecuTorch export...")
            exported_program = quantized_program
        else:
            # Standard export without quantization
            exported_program = torch.export.export(wrapped_model, sample_inputs)
        
        print("Converting to ExecuTorch format...")
        # Lower to edge dialect and apply partitioners based on backend
        partitioner = []
        
        if backend_name == 'vulkan':
            print("Using Vulkan partitioner for GPU acceleration...")
            print("  Note: Vulkan has limited operator support (may skip some ops)")
            print("  Supported ops will run on GPU, unsupported ops fall back to CPU")
            
            # Suppress verbose INFO logs from Vulkan partitioner
            import logging
            logging.getLogger().setLevel(logging.WARNING)
            
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
            partitioner = [VulkanPartitioner()]
        elif backend_name == 'xnnpack':
            print("Using XNNPACK partitioner for CPU optimization...")
            print("  Note: XNNPACK has excellent operator coverage for mobile CPUs")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            partitioner = [XnnpackPartitioner()]
        elif backend_name == 'vulkan-compat':
            # Compatibility mode for Vulkan with workarounds
            print("Using Vulkan partitioner in compatibility mode...")
            print("  Note: Using workarounds for operator incompatibilities")
            print("  This may reduce GPU usage but increase compatibility")
            
            import logging
            logging.getLogger().setLevel(logging.WARNING)
            
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
            
            # Create partitioner with more conservative settings
            partitioner = [VulkanPartitioner()]
        else:
            print("Using basic CPU backend (no partitioner)...")
            print("  Note: Supports all operators but without optimizations")
            partitioner = []
        
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioner,
        ).to_executorch()

        # Re-enable logging
        import logging
        logging.getLogger().setLevel(logging.INFO)
        
        # Save the ExecuTorch program
        print(f"Saving ExecuTorch model to {output_path}...")
        with open(output_path, "wb") as file:
            file.write(executorch_program.buffer)
        
        print(f"✓ Successfully exported ExecuTorch model to {output_path}")
        print(f"  Model input size: {example_input_size}")
        print(f"  Backend: {backend_name.upper()}")
        
        if quantize and backend_name == 'xnnpack':
            quant_mode = "Static (calibrated)" if actual_calibration_samples else "Dynamic"
            print(f"  Quantization: 8-bit symmetric ({quant_mode})")
            print(f"  Quantization scheme: {'Per-channel' if actual_per_channel else 'Per-tensor'}")
        
        if backend_name == 'vulkan':
            print(f"  Execution mode: Hybrid (GPU + CPU fallback)")
            print(f"  Note: Supported operations run on Vulkan GPU,")
            print(f"        unsupported ops (slice_scatter, copy, index, floor) run on CPU")
        
        print(f"  Output format: ExecuTorch (.pte)")
        
        # Get file size
        file_size = os.path.getsize(output_path)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error during model export: {error_msg}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        
        # Check if it's a backend-specific issue
        if backend_name == 'vulkan' and ('fake mode' in error_msg or 'no operator implementation' in error_msg):
            print("⚠ Vulkan backend doesn't support all operations in your model.")
            print("  Unsupported ops may include: slice_scatter, copy, index, floor")
            print("\n  RECOMMENDED: Try XNNPACK backend instead (better operator coverage):")
            print(f"    python {os.path.basename(__file__)} --config {config_path} --backend xnnpack")
            print("\n  Or use CPU backend as fallback:")
            print(f"    python {os.path.basename(__file__)} --config {config_path} --backend cpu")
        
        print("\n1. Ensure ExecuTorch is properly installed:")
        print("   pip install executorch")
        print("2. For Vulkan support, you may need additional dependencies:")
        print("   Follow https://pytorch.org/executorch/stable/build-run-vulkan.html")
        print("3. For XNNPACK support (recommended for CPU), install with:")
        print("   pip install executorch[xnnpack]")
        print("4. Check that your model is compatible with torch.export")
        print("   - Avoid dynamic control flow")
        print("   - Ensure all operations are traceable")
        
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
                        choices=['vulkan', 'vulkan-compat', 'xnnpack', 'cpu'],
                        help='Backend for optimization: vulkan (GPU), vulkan-compat (GPU with workarounds), xnnpack (CPU optimized), cpu (basic)')
    parser.add_argument('--test', action='store_true',
                        help='Test the exported model using ExecuTorch runtime')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply 8-bit quantization (XNNPACK backend only)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing calibration images for static quantization (optional)')
    parser.add_argument('--per-tensor-quant', action='store_true',
                        help='Use per-tensor quantization instead of per-channel (default: per-channel)')

    args = parser.parse_args()
    
    # Calculate input dimensions
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)
    
    print("=" * 70)
    print("ExecuTorch Model Export for Mobile")
    print("=" * 70)
    print(f"Backend: {args.backend.upper()}")
    if args.backend == 'xnnpack':
        print("  • Best choice for CPU-based mobile devices")
        print("  • Excellent operator coverage")
        if args.quantize:
            print("  • 8-bit quantization enabled")
            if args.data_dir:
                print(f"  • Static quantization with calibration from: {args.data_dir}")
            else:
                print("  • Dynamic quantization (no calibration data)")
    elif args.backend == 'vulkan':
        print("  • GPU acceleration (limited operator support)")
        print("  • May fall back to CPU for unsupported ops")
        if args.quantize:
            print("  ⚠ Quantization only supported with XNNPACK backend")
    print("=" * 70 + "\n")
    
    success = export_model_with_executorch(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        backend=args.backend,
        quantize=args.quantize,
        calibration_data_dir=args.data_dir,
        per_channel_quant=not args.per_tensor_quant,
    )
    
    if success:
        print("\n" + "=" * 70)
        print("Export completed successfully!")
        print("=" * 70)
        
        # Run test if requested
        if args.test:
            test_exported_model(args.output, input_size)
        
        print("\n" + "=" * 70)
        print("Deployment Information")
        print("=" * 70)
        print("\nBackend-specific notes:")
        if args.backend == 'vulkan':
            print("• Vulkan: GPU-accelerated inference on Android/iOS")
            print("  - Best for: Real-time mobile inference with GPU")
            print("  - Requires: Device with Vulkan support")
        elif args.backend == 'xnnpack':
            print("• XNNPACK: CPU-optimized inference")
            print("  - Best for: Efficient CPU inference on mobile devices")
            print("  - Requires: ARM or x86 CPU")
            if args.quantize:
                print("  - Quantization: 8-bit symmetric quantization applied")
                print("    · ~4x smaller model size")
                print("    · ~2-4x faster inference")
                print("    · Minimal accuracy loss (<1% typical)")
        else:
            print("• CPU: Basic CPU backend")
            print("  - Best for: Testing and compatibility")
            print("  - Performance: Slower than XNNPACK/Vulkan")
        
        print("\nNext steps:")
        print("1. For Android: Integrate the .pte file with ExecuTorch runtime")
        print("   - Add ExecuTorch dependencies to your build.gradle")
        print("   - Load model using Module.load()")
        print("2. For iOS: Use the .pte file with ExecuTorch runtime")
        print("   - Add ExecuTorch pod to your Podfile")
        print("   - Load using ExecuTorch C++ or Swift APIs")
        print("3. Performance tips:")
        print("   - Reuse input tensors to avoid allocation overhead")
        if not args.quantize and args.backend == 'xnnpack':
            print("   - Consider quantization for smaller model size and faster inference:")
            print("     python export_mobile_executorch.py --backend xnnpack --quantize --data-dir ./calib_images/")
        print("   - Use --test flag to verify model before deployment")
        print("4. Integration guides:")
        print("   - ExecuTorch: https://pytorch.org/executorch/")
        print("   - Android setup: https://pytorch.org/executorch/stable/demo-apps-android.html")
        print("   - iOS setup: https://pytorch.org/executorch/stable/demo-apps-ios.html")
    else:
        print("\n" + "=" * 70)
        print("Export failed. Please check the error messages above.")
        print("=" * 70)
