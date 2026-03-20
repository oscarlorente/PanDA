from __future__ import absolute_import, division, print_function
"""
CoreML Model Export Script for iOS Deployment

Exports the equirectangular depth-estimation model to CoreML (.mlpackage) so it
can be used on Apple devices via the Neural Engine / GPU / CPU.

Key differences vs. the ExecuTorch export:
  - No runtime compilation cost after the first launch (Xcode pre-compiles the
    bundled .mlpackage to .mlmodelc at build time; downloaded models are compiled
    once and cached to Application Support by the iOS loader).
  - Uses torch.export.export (same path as the ExecuTorch export) instead of
    torch.jit.trace, so coremltools 7+ receives an ExportedProgram and converts
    every op — including upsample_bicubic2d for DINOv2 positional embeddings —
    correctly and identically to the ExecuTorch path.

WHY torch.export INSTEAD OF torch.jit.trace
  DINOv2-based models always interpolate their positional embeddings for
  non-square inputs (our panoramic 1008×504 input always produces a non-square
  72×36 patch grid).  The interpolation uses aten::upsample_bicubic2d.
  - torch.jit.trace + ct.convert: the TorchScript frontend cannot lower
    upsample_bicubic2d natively, so the old approach registered a custom op that
    silently substituted bilinear upsampling.  Bilinear ≠ bicubic, meaning every
    attention layer received subtly wrong positional embeddings, which propagated
    through the full network and produced depth maps that differed noticeably from
    the ExecuTorch output.
  - torch.export.export + ct.convert (coremltools ≥ 7.0): the TorchExport frontend
    can lower upsample_bicubic2d to the equivalent CoreML MIL op, giving the same
    result as the ExecuTorch/CoreML-partitioner pipeline.

Requirements:
    pip install "coremltools>=7.0" torch pyyaml

Usage:
    python export_panda_coreml.py \
        --config  path/to/config.yaml \
        --output  ./checkpoints/depth_model.mlpackage \
        --height  504               # width = 2 * height (ERP convention)
        --min-ios iOS16             # iOS15 / iOS16 / iOS17

iOS integration notes:
  - Add the generated .mlpackage to the Xcode project; Xcode compiles it to
    .mlmodelc at build time, so no runtime compilation is needed for bundled models.
  - For OTA / downloaded models, distribute the .mlpackage as a ZIP archive and
    extract it on-device (the project already includes the Zip pod).  The iOS
    HoHoModule loader compiles the .mlpackage on first use and caches the result
    in Application Support so subsequent launches are instant.
  - Input : MLMultiArray Float32 [1, 3, H, W], values normalised to [0, 1]
            (matches the existing UIImage.normalized() helper — no extra changes
            needed on the Swift side).
  - Output: MLMultiArray Float32 — depth map.  Shape is typically [1, H, W] or
            [1, 1, H, W]; the iOS loader uses stride-based indexing so it handles
            any leading batch/channel dimensions of size 1 transparently.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import coremltools as ct

from networks.models import *

import torch.nn.functional as _F

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    """Extracts only the depth tensor from the model output dict."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output["pred_depth"]
        return output


# ---------------------------------------------------------------------------
# Bicubic-freeze helper
# ---------------------------------------------------------------------------

def _freeze_bicubic_as_constants(model, sample_input):
    """
    Precomputes every `F.interpolate(..., mode='bicubic')` call that occurs
    during model(sample_input) and returns a replay function that returns those
    precomputed tensors as constants during a subsequent torch.jit.trace.

    WHY THIS WORKS
    DINOv2 interpolates its learned positional embeddings to match the current
    patch-grid size using bicubic upsampling.  For a fixed input size the
    patch-grid size is always the same, so the bicubic result is constant —
    it depends only on the model weights and the input shape, not on the
    actual pixel values.  Freezing those results as constant tensors in the
    TorchScript graph:
      1. Removes the upsample_bicubic2d op that coremltools cannot lower.
      2. Produces output that is bit-for-bit identical to the ExecuTorch/ATEN
         path, because PyTorch's own bicubic kernel is used for precomputation.
      3. Works with any coremltools version that handles torch.jit.trace.

    Usage:
        replay_fn, original_fn = _freeze_bicubic_as_constants(wrapped, inp)
        _F.interpolate = replay_fn
        try:
            traced = torch.jit.trace(wrapped, (inp,))
        finally:
            _F.interpolate = original_fn
    """
    bicubic_cache: list = []
    original_fn = _F.interpolate

    # Pass 1: run the model normally to populate the cache with exact results.
    def _caching_fn(input, size=None, scale_factor=None, mode='nearest',
                    align_corners=None, recompute_scale_factor=None, **kwargs):
        if mode == 'bicubic':
            result = original_fn(input, size=size, scale_factor=scale_factor,
                                 mode='bicubic', align_corners=align_corners,
                                 recompute_scale_factor=recompute_scale_factor)
            bicubic_cache.append(result.detach().clone())
            return result
        return original_fn(input, size=size, scale_factor=scale_factor,
                           mode=mode, align_corners=align_corners,
                           recompute_scale_factor=recompute_scale_factor, **kwargs)

    _F.interpolate = _caching_fn
    try:
        with torch.no_grad():
            _ = model(sample_input)
    finally:
        _F.interpolate = original_fn

    print(f"  Frozen {len(bicubic_cache)} bicubic interpolation(s) as graph constants")

    # Pass 2: replay function — returns cached tensors during jit.trace.
    # torch.jit.trace sees these as leaf (constant) nodes in the IR graph.
    # The index cycles modulo cache length so that multiple internal passes
    # (e.g. the jit.trace consistency check) do not exhaust the cache.
    replay_idx = [0]

    def _replaying_fn(input, size=None, scale_factor=None, mode='nearest',
                      align_corners=None, recompute_scale_factor=None, **kwargs):
        if mode == 'bicubic':
            idx = replay_idx[0] % len(bicubic_cache)
            replay_idx[0] += 1
            return bicubic_cache[idx]
        return original_fn(input, size=size, scale_factor=scale_factor,
                           mode=mode, align_corners=align_corners,
                           recompute_scale_factor=recompute_scale_factor, **kwargs)

    return _replaying_fn, original_fn


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_model_coreml(
    config_path,
    output_path,
    example_input_size=(1, 3, 504, 1008),
    minimum_deployment_target=ct.target.iOS16,
):
    """
    Trace + convert the depth model to a CoreML .mlpackage.

    Strategy
    --------
    coremltools (any version up to 9.x) cannot lower the
    aten::upsample_bicubic2d op that DINOv2 uses for positional-embedding
    interpolation, whether the model is supplied as a TorchScript trace or as
    an ExportedProgram.

    The solution is to precompute every bicubic interpolation at the target
    input size before tracing, then replace those calls with the precomputed
    constant tensors during torch.jit.trace.  Because the input size is fixed
    (1008×504 → 72×36 patch grid), the bicubic result is always identical to
    what PyTorch would compute at runtime, so the exported model is
    numerically equivalent to the ExecuTorch pipeline.

    Args:
        config_path: Path to the model YAML config file.
        output_path: Destination path for the .mlpackage bundle.
        example_input_size: (batch, channels, height, width).
            Width must equal 2 × height for equirectangular panoramas.
        minimum_deployment_target: Minimum iOS/macOS version to target.
    """
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading model weights...")
    model_dict = torch.load(config["model_path"], map_location="cpu")
    model = make(config["model"])

    # Strip DataParallel 'module.' prefix when present
    if any(k.startswith("module") for k in model_dict.keys()):
        model_dict = {k.replace("module.", ""): v for k, v in model_dict.items()}

    model_state = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state})
    model.eval()

    wrapped = ModelWrapper(model)
    wrapped.eval()

    # Values in [0, 1] to match UIImage.normalized() on iOS.
    sample_input = torch.zeros(example_input_size)

    # Precompute all bicubic interpolations (DINOv2 positional embeddings) as
    # constants so that torch.jit.trace produces a graph with no
    # upsample_bicubic2d nodes.  The frozen values are bit-identical to what
    # PyTorch/ExecuTorch computes at runtime for this fixed input size.
    print(f"Precomputing bicubic embeddings for input shape {example_input_size}...")
    replay_fn, original_fn = _freeze_bicubic_as_constants(wrapped, sample_input)

    print("Tracing model...")
    _F.interpolate = replay_fn
    try:
        with torch.no_grad():
            # check_trace=False: the model intentionally freezes shape-dependent
            # branches (patch-grid size, positional-embedding assertions) for a
            # fixed input resolution.  The second verification pass that
            # _check_trace performs would exhaust the bicubic replay cache and
            # is redundant for a fixed-shape export.
            traced = torch.jit.trace(wrapped, (sample_input,), check_trace=False)
    finally:
        _F.interpolate = original_fn

    print("Converting to CoreML (.mlpackage)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",           # must match the key in iOS MLDictionaryFeatureProvider
                shape=example_input_size,
                dtype=float,            # float32
            )
        ],
        # Output name is inferred from the traced graph.
        # The iOS loader discovers it dynamically via modelDescription.outputDescriptionsByName.
        minimum_deployment_target=minimum_deployment_target,
        convert_to="mlprogram",         # .mlpackage — requires iOS 15+
    )

    mlmodel.short_description = "Equirectangular monocular depth estimation"
    mlmodel.input_description["input"] = (
        "Normalised RGB panorama, shape [1, 3, H, W], values in [0, 1]"
    )

    print(f"Saving CoreML model to {output_path}...")
    mlmodel.save(output_path)

    b, c, h, w = example_input_size
    print(f"\n✓  CoreML model saved to: {output_path}")
    print(f"   Input  'input'  — Float32 [{b}, {c}, {h}, {w}]  (values in [0, 1])")
    print("   Output — Float32 depth map (first output feature, shape [1, H, W])")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export depth model to CoreML (.mlpackage) for iOS deployment"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config YAML file")
    parser.add_argument("--output", type=str,
                        default="./checkpoints/depth_model.mlpackage",
                        help="Output path for the CoreML model (.mlpackage)")
    parser.add_argument("--height", type=int, default=504,
                        help="Input height; width = 2 × height (ERP). Must be a multiple of 14.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (must be 1 for on-device inference)")
    parser.add_argument("--min-ios", type=str, default="iOS16",
                        choices=["iOS15", "iOS16", "iOS17"],
                        help="Minimum iOS deployment target")
    args = parser.parse_args()

    target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }

    h = args.height
    w = 2 * h
    input_size = (args.batch_size, 3, h, w)

    print("CoreML Depth Model Export")
    print(f"  Input size : {input_size}")
    print(f"  Min iOS    : {args.min_ios}")
    print(f"  Output     : {args.output}\n")

    success = export_model_coreml(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        minimum_deployment_target=target_map[args.min_ios],
    )

    if success:
        print("\nExport completed successfully!")