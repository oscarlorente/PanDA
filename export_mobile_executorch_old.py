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

FP16 Export Mode:
    Model weights are FP16, but I/O stays FP32.
    The graph contains explicit fp32->fp16 and fp16->fp32 casts at the
    boundary so that backend runs all internal ops in FP16 while Android
    passes/receives plain float[] tensors (no change needed on Android side).

If Vulkan export fails, try XNNPACK instead.
"""
import argparse
import yaml
import torch
import torch.nn as nn

from executorch.exir import to_edge_transform_and_lower

from networks.models import *


class ModelWrapperFP16(nn.Module):
    """
    Wrapper for FP16 export.

    Keeps the graph's external I/O in FP32 (so Android sends/receives float[])
    but performs all internal computation in FP16 by casting at the boundary.
    This is required because ExecuTorch's edge dialect verifier enforces that
    all tensor arguments to each op share the same dtype, so we cannot have
    a mix of FP32 I/O and FP16 weights at the graph boundary — the cast ops
    make the types consistent throughout.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.half()                        # FP32 -> FP16 at graph entry
        output = self.model(x)
        if isinstance(output, dict):
            return output['pred_depth'].float()   # FP16 -> FP32 at graph exit
        return output.float()


class ModelWrapperFP32(nn.Module):
    """
    Standard wrapper for FP32 export or true end-to-end FP16 export.

    For FP32: no casts, I/O and weights are all float32.
    For true FP16 (non-Vulkan): model is converted to .half() before export
    and sample_inputs are also .half(), so the entire graph is FP16 including
    I/O — Android must supply a FP16 ShortBuffer and decode FP16 output.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output['pred_depth']
        return output


def export_model_with_executorch(
    config_path,
    output_path,
    example_input_size=(1, 3, 504, 1008),
    backend='vulkan',
    use_fp16=False,
):
    """
    Export model using ExecuTorch for mobile deployment.

    Args:
        config_path: Path to config YAML file
        output_path: Path where exported model will be saved (.pte)
        example_input_size: Tuple of (batch, channels, height, width)
        backend: 'vulkan' | 'xnnpack' | 'coreml' | 'cpu'
        use_fp16: If True, export in FP16. Behaviour differs by backend:
                  - vulkan:  FP32 I/O with internal FP16 (no Android changes needed)
                  - others:  True end-to-end FP16 (Android must use FP16 tensors)
    """
    backend_name = backend.lower()

    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading model...")
    model_dict = torch.load(config["model_path"], map_location='cpu')

    model = make(config['model'])

    if any(key.startswith('module') for key in model_dict.keys()):
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

    model_state_dict = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in model_dict.items() if k in model_state_dict}
    )
    model.eval()

    # --- Choose wrapper and precision ---
    if use_fp16:
        print("FP16 mode: FP32 I/O with internal FP16 casts.")
        print("  Android side: no changes needed, use plain float[] tensors.")
        wrapped_model = ModelWrapperFP16(model).eval()
        wrapped_model = wrapped_model.half()   # all weights -> FP16
        wrapped_model.use_fp16 = True          # keep flag consistent
        sample_inputs = (torch.randn(example_input_size),)   # FP32 input

    else:
        print("FP32 mode.")
        wrapped_model = ModelWrapperFP32(model).eval()
        sample_inputs = (torch.randn(example_input_size),)   # FP32 input

    precision_label = (
        "FP16 (Vulkan, FP32 I/O)" if use_fp16
        else "FP32"
    )
    print(f"Model loaded. Precision: {precision_label}")
    print(f"Exporting with input size {example_input_size}...")

    try:
        print("Running torch.export.export()...")
        exported_program = torch.export.export(wrapped_model, sample_inputs)

        print("Converting to ExecuTorch format...")
        partitioner = []
        edge_compile_config = None

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
            from executorch.exir import EdgeCompileConfig
            partitioner = [CoreMLPartitioner()]
            edge_compile_config = EdgeCompileConfig(_skip_dim_order=True)

        else:
            print("Using basic CPU backend (no partitioner)...")
            partitioner = []

        to_edge_kwargs = dict(partitioner=partitioner)
        if edge_compile_config is not None:
            to_edge_kwargs['compile_config'] = edge_compile_config

        # After to_edge_transform_and_lower, before .to_executorch():
        edge_program = to_edge_transform_and_lower(exported_program, **to_edge_kwargs)

        # Check how many nodes were actually delegated to XNNPACK
        graph = edge_program.exported_program().graph
        total_nodes = sum(1 for n in graph.nodes if n.op == 'call_function')
        delegated_nodes = sum(1 for n in graph.nodes if 'lowered_module' in str(n))
        print(f"Graph nodes: {total_nodes} total, {delegated_nodes} delegated to backend")

        executorch_program = edge_program.to_executorch()

        print(f"Saving ExecuTorch model to {output_path}...")
        with open(output_path, "wb") as file:
            file.write(executorch_program.buffer)

        print(f"✓ Successfully exported ExecuTorch model to {output_path}")
        print(f"  Input size : {example_input_size}")
        print(f"  Backend    : {backend_name.upper()}")
        print(f"  Precision  : {precision_label}")

        return True

    except Exception as e:
        print(f"✗ Error during model export: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export model using ExecuTorch for mobile deployment'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output', type=str,
                        default='./checkpoints/model_mobile.pte',
                        help='Output path for ExecuTorch model (.pte)')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width = 2x height). Must be a multiple of 14.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for example input')
    parser.add_argument('--backend', type=str, default='vulkan',
                        choices=['vulkan', 'xnnpack', 'coreml', 'cpu'],
                        help=(
                            'Backend: vulkan (GPU), xnnpack (CPU optimized), '
                            'coreml (Apple Neural Engine), cpu (basic)'
                        ))
    parser.add_argument('--fp16', action='store_true',
                        help=(
                            'Export in FP16. FP32 I/O, internal FP16. No Android changes needed.'
                        ))

    args = parser.parse_args()

    erp_height = args.height
    erp_width  = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)

    print("ExecuTorch Model Export for Mobile")
    success = export_model_with_executorch(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        backend=args.backend,
        use_fp16=args.fp16,
    )

    if success:
        print("Export completed successfully!")