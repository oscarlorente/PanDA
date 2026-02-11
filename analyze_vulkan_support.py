#!/usr/bin/env python3
"""
Analyze which operators in your model are supported by Vulkan backend.

This script exports your model and analyzes the Vulkan partitioner logs
to show exactly which operations will run on GPU vs CPU.
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import logging
from collections import defaultdict

from networks.models import *


class ModelWrapper(nn.Module):
    """Wrapper to extract only the depth prediction from the model output."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output['pred_depth']
        return output


class VulkanAnalysisHandler(logging.Handler):
    """Custom logging handler to capture Vulkan partitioner decisions."""
    
    def __init__(self):
        super().__init__()
        self.skipped_ops = defaultdict(int)
        self.partitioned_subgraphs = 0
        self.reasons = defaultdict(int)
        
    def emit(self, record):
        msg = record.getMessage()
        
        # Capture skipped operations
        if "skipping" in msg and "Due to" in msg:
            # Extract operation name
            if "aten." in msg:
                op_start = msg.find("aten.")
                op_end = msg.find("(", op_start)
                if op_end > op_start:
                    op_name = msg[op_start:op_end].strip()
                    self.skipped_ops[op_name] += 1
                    
                # Extract reason
                reason_start = msg.find("[") + 1
                reason_end = msg.find("]", reason_start)
                if reason_end > reason_start:
                    reason = msg[reason_start:reason_end]
                    self.reasons[reason] += 1
        
        # Capture successful partitions
        if "Vulkan subgraphs to be partitioned" in msg:
            import re
            match = re.search(r'Found (\d+) Vulkan subgraphs', msg)
            if match:
                self.partitioned_subgraphs = int(match.group(1))


def analyze_vulkan_support(config_path, input_size=(1, 3, 504, 1008)):
    """
    Analyze Vulkan backend support for the model.
    
    Returns statistics about which operations are supported.
    """
    print("=" * 80)
    print("Vulkan Backend Support Analysis")
    print("=" * 80)
    
    # Load model
    print(f"\n[1/4] Loading model from config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_dict = torch.load(config["model_path"], map_location='cpu')
    model = make(config['model'])
    
    if any(key.startswith('module') for key in model_dict.keys()):
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
    
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Export model
    print(f"\n[2/4] Exporting model with torch.export...")
    sample_inputs = (torch.randn(input_size),)
    
    try:
        exported_program = torch.export.export(wrapped_model, sample_inputs)
        print(f"✓ Export successful ({len(exported_program.graph.nodes)} graph nodes)")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return
    
    # Analyze with Vulkan partitioner
    print(f"\n[3/4] Analyzing Vulkan backend compatibility...")
    
    # Set up custom logging to capture partition decisions
    handler = VulkanAnalysisHandler()
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        
        _ = to_edge_transform_and_lower(
            exported_program,
            partitioner=[VulkanPartitioner()],
        ).to_executorch()
        
    except Exception as e:
        print(f"⚠ Partitioning completed with errors: {type(e).__name__}")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
    
    # Display results
    print(f"\n[4/4] Analysis Results")
    print("=" * 80)
    
    print(f"\n✓ GPU-Accelerated Subgraphs: {handler.partitioned_subgraphs}")
    if handler.partitioned_subgraphs > 0:
        print(f"  These subgraphs will execute on Vulkan GPU")
        print(f"  Common ops: conv2d, linear, matmul, add, mul, relu, gelu, etc.")
    
    print(f"\n✗ Unsupported Operations (CPU Fallback): {len(handler.skipped_ops)} unique ops")
    if handler.skipped_ops:
        print(f"\n  Top unsupported operations:")
        for op, count in sorted(handler.skipped_ops.items(), key=lambda x: -x[1])[:10]:
            print(f"    • {op}: {count} occurrences")
    
    print(f"\n  Reasons for skipping:")
    for reason, count in sorted(handler.reasons.items(), key=lambda x: -x[1]):
        print(f"    • {reason}: {count} operations")
    
    # Calculate approximate GPU usage
    total_ops = sum(handler.skipped_ops.values())
    print(f"\n📊 Estimated GPU Utilization:")
    if handler.partitioned_subgraphs > 0:
        print(f"  • GPU subgraphs: {handler.partitioned_subgraphs}")
        print(f"  • CPU fallback ops: {total_ops}")
        print(f"  • Execution mode: HYBRID (GPU + CPU)")
        print(f"\n  ✓ Model CAN benefit from Vulkan GPU acceleration")
        print(f"  ✓ Majority of compute-heavy ops likely on GPU")
        print(f"  ⚠ Some operations will fall back to CPU")
    else:
        print(f"  ✗ No GPU acceleration possible")
        print(f"  Recommendation: Use XNNPACK backend instead")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    
    if handler.partitioned_subgraphs > 20:
        print("✓ GOOD: Vulkan backend can accelerate significant portions of your model")
        print("  • Try exporting: python export_mobile_executorch.py --backend vulkan")
        print("  • Expect hybrid GPU+CPU execution")
    elif handler.partitioned_subgraphs > 0:
        print("⚠ MODERATE: Some GPU acceleration available but limited")
        print("  • Vulkan may provide modest speedup")
        print("  • Consider: python export_mobile_executorch.py --backend xnnpack")
    else:
        print("✗ POOR: Vulkan cannot accelerate this model")
        print("  • Recommended: python export_mobile_executorch.py --backend xnnpack")
    
    print("\n💡 Key Insights:")
    print("  • Vulkan partitioner creates HYBRID execution plans")
    print("  • Supported ops → GPU (fast)")
    print("  • Unsupported ops → CPU fallback (automatic)")
    print("  • Model will still run, just not 100% on GPU")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze Vulkan backend support for your model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default config
  python analyze_vulkan_support.py --config config/inference/custom_vits.yaml
  
  # Analyze with custom input size
  python analyze_vulkan_support.py --config config/inference/custom_vitb.yaml --height 252
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width will be 2x height)')

    args = parser.parse_args()
    
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (1, 3, erp_height, erp_width)
    
    analyze_vulkan_support(args.config, input_size)
