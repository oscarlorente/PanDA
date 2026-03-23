from __future__ import absolute_import, division, print_function
"""
ONNX Model Export Script for Mobile Deployment

Optimization flags:
  --merge-lora          Merge LoRA A/B matrices into base QKV weights (default on).
                        Zero accuracy change, eliminates LoRA runtime overhead.

  --fuse-layer-scale    Fuse LayerScale gamma into attn.proj and mlp.fc2 weights
                        (default on). Zero accuracy change, eliminates 24 element-
                        wise multiplications per forward pass (2 per ViT block × 12).

  --fuse-bn             Fuse all Conv2d+BatchNorm2d pairs into a single Conv2d
                        (default on). Zero accuracy change. Covers ConvBlock in
                        dpt.py and any ResidualConvUnit built with use_bn=True.
                        Applied in PyTorch before export so the BN nodes never
                        appear in the ONNX graph at all.

  --static-interp       Pre-compute all bilinear interpolation target sizes from
                        the fixed input shape and bake them into FeatureFusionBlock
                        instances before tracing (default on). Eliminates all
                        dynamic shape computations in Resize ONNX nodes, which
                        lets the runtime constant-fold the entire decoder upsample
                        chain at load time.

  --onnx-sim            Run onnx-simplifier on the exported graph (default on).
                        Constant-folds, removes dead nodes, simplifies shapes.
                        Requires: pip install onnxsim

  --ort-optimize        Run OnnxRuntime's offline graph optimizer after export
                        (default on). Fuses remaining BN/Gelu/LayerNorm patterns,
                        eliminates redundant Cast nodes, and runs attention fusion
                        heuristics. Produces a separate *_opt.onnx file.
                        Requires: pip install onnxruntime

  --encoder-layers 2    Use only 2 intermediate encoder layers instead of 4.
                        Model always loads from the full 4-layer checkpoint.
                        Expect ~20-30% encoder speedup with some quality loss.
                        Evaluate quality before using in production.

  --fp16                Export with FP16 weights. Uses FP32 I/O with internal FP16
                        so no runtime code changes are needed.
"""
import argparse
import copy
import os
import yaml
import torch
import torch.nn as nn
import torch.onnx


from networks.models import *


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Optimization passes (applied to the PyTorch model before tracing)
# ---------------------------------------------------------------------------

def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """
    Fuse every Conv2d → BatchNorm2d pair in the model in-place.

    torch.nn.utils.fusion.fuse_conv_bn_eval handles the weight math:
        W_fused = W * (gamma / sqrt(var + eps))
        b_fused = (b - mean) * gamma / sqrt(var + eps) + beta

    This covers:
      - ConvBlock.conv_block  (Conv2d → BN2d → ReLU in dpt.py)
      - ResidualConvUnit.conv1/conv2 + bn1/bn2  (when use_bn=True in blocks.py)

    The model must already be in eval() mode before calling this.
    BN nodes are replaced with nn.Identity() so the module structure is
    preserved (no index shifts in nn.Sequential).

    Args:
        model: a PanDA instance in eval() mode.

    Returns:
        model (in-place modified)
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    fused_count = 0

    # --- Pass 1: fuse nn.Sequential children that are Conv2d followed by BN ---
    # Covers ConvBlock.conv_block and any other Sequential with this pattern.
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        i = 0
        while i < len(children) - 1:
            name_a, layer_a = children[i]
            name_b, layer_b = children[i + 1]
            if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
                fused = fuse_conv_bn_eval(layer_a, layer_b)
                setattr(module, name_a, fused)
                setattr(module, name_b, nn.Identity())
                fused_count += 1
                i += 2  # skip the BN we just consumed
            else:
                i += 1

    # --- Pass 2: fuse ResidualConvUnit.conv{1,2} + bn{1,2} pairs ---
    # These are not sequential — the BN is a sibling attribute applied
    # manually in forward(), so we need to handle them explicitly.
    #
    # ResidualConvUnit.forward() when self.bn=True:
    #   out = conv1(out)
    #   out = bn1(out)       ← fuse bn1 into conv1
    #   out = conv2(out)
    #   out = bn2(out)       ← fuse bn2 into conv2
    for module in model.modules():
        # Duck-type check: has conv1+bn1 and conv2+bn2 and the .bn flag
        if not (hasattr(module, 'bn') and module.bn
                and hasattr(module, 'conv1') and isinstance(module.conv1, nn.Conv2d)
                and hasattr(module, 'bn1')   and isinstance(module.bn1, nn.BatchNorm2d)
                and hasattr(module, 'conv2') and isinstance(module.conv2, nn.Conv2d)
                and hasattr(module, 'bn2')   and isinstance(module.bn2, nn.BatchNorm2d)):
            continue

        module.conv1 = fuse_conv_bn_eval(module.conv1, module.bn1)
        module.bn1   = nn.Identity()
        module.conv2 = fuse_conv_bn_eval(module.conv2, module.bn2)
        module.bn2   = nn.Identity()
        # Patch forward so the now-Identity BN calls are no-ops.
        # The easiest way is to clear the flag so the if-branches are skipped.
        module.bn = False
        fused_count += 1

    print(f"fuse_conv_bn: fused {fused_count} Conv+BN pair(s) into Conv2d.")
    return model


def patch_static_interpolation_sizes(model: nn.Module, input_h: int, input_w: int) -> nn.Module:
    """
    Pre-compute all bilinear upsample target sizes in the DPT decoder and
    bake them into the FeatureFusionBlock instances before ONNX tracing.

    Why this matters
    ----------------
    FeatureFusionBlock.forward() chooses between three interpolation modes:
        (a) scale_factor=2          — when no size is given at all
        (b) size=self.size          — when a static size was set at construction
        (c) size=<runtime tensor>   — when the caller passes size=x.shape[2:]

    Cases (a) and (c) produce ONNX Resize nodes whose output-size input is a
    runtime tensor. Mobile runtimes (NNAPI, QNN, CoreML) cannot constant-fold
    these at load time, so they re-compute shapes on every inference call.
    Case (b) produces a Resize node with a fully static sizes input, which the
    runtime can fold into the operator parameters once and never revisit.

    What we compute
    ---------------
    For a fixed input of shape (1, 3, H, W), the ViT patch grid is:
        patch_h = H // 14,  patch_w = W // 14

    The DPT decoder processes features at these spatial scales (for 4-layer mode):
        layer_1 after resize_layers[0] (ConvTranspose2d ×4): (patch_h*4,  patch_w*4)
        layer_2 after resize_layers[1] (ConvTranspose2d ×2): (patch_h*2,  patch_w*2)
        layer_3 after resize_layers[2] (Identity):           (patch_h,    patch_w)
        layer_4 after resize_layers[3] (Conv2d stride=2):    (patch_h//2, patch_w//2)

    The refinenet fusion chain then upsamples toward the highest resolution:
        refinenet4 output → upsample to layer_3 size:  (patch_h,   patch_w)
        refinenet3 output → upsample to layer_2 size:  (patch_h*2, patch_w*2)
        refinenet2 output → upsample to layer_1 size:  (patch_h*4, patch_w*4)
        refinenet1 output → upsample ×2 (no target):   (patch_h*8, patch_w*8)

    The final output_conv1 interpolate in DPTHead.forward targets:
        (patch_h * 14, patch_w * 14)  ==  (H, W)

    We set self.size on each FeatureFusionBlock so that the (a) and (c)
    branches are never taken during tracing. The DPTHead.forward call-sites
    that pass explicit size= kwargs override self.size anyway — those are
    already static tuples derived from .shape[2:] of already-computed tensors,
    which the tracer resolves to concrete integers. The only truly dynamic
    case is refinenet1, which falls through to scale_factor=2; we patch that
    one too.

    This function is safe to call for both 2-layer and 4-layer encoder modes
    because FeatureFusionBlock.size is only used when the caller does NOT pass
    an explicit size= kwarg. In 4-layer mode refinenet1 is the only block that
    receives no size kwarg; in 2-layer mode both refinenet3 and refinenet4 are
    called without an explicit size at the second call site.

    Args:
        model:   a PanDA instance whose model.core is a DepthAnythingV2.
        input_h: the fixed input height (e.g. 504).
        input_w: the fixed input width  (e.g. 1008).

    Returns:
        model (in-place modified)
    """
    if not (hasattr(model, 'core') and hasattr(model.core, 'depth_head')):
        print("patch_static_interpolation_sizes: could not find model.core.depth_head — skipping.")
        return model

    depth_head = model.core.depth_head
    scratch     = depth_head.scratch

    patch_h = input_h // 14
    patch_w = input_w // 14

    # Spatial sizes at each decoder stage (height, width)
    # These mirror the shapes that arrive at each refinenet during forward().
    size_l4 = (patch_h // 2, patch_w // 2)   # after resize_layers[3] stride-2 conv
    size_l3 = (patch_h,      patch_w)         # after resize_layers[2] identity
    size_l2 = (patch_h * 2,  patch_w * 2)     # after resize_layers[1] ×2 deconv
    size_l1 = (patch_h * 4,  patch_w * 4)     # after resize_layers[0] ×4 deconv

    # refinenet4: upsamples from size_l4 → size_l3
    #   Called as: refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    #   The explicit size= kwarg always wins over self.size, so this is already
    #   a static tuple at trace time. Setting self.size here is a safety net
    #   for the 2-layer path where refinenet4 may be called without size=.
    scratch.refinenet4.size = size_l3

    # refinenet3: upsamples from size_l3 → size_l2
    #   4-layer path: called with explicit size=layer_2_rn.shape[2:] → static
    #   2-layer path: called WITHOUT size= kwarg → falls back to self.size
    scratch.refinenet3.size = size_l2

    # refinenet2: upsamples from size_l2 → size_l1
    #   Called with explicit size=layer_1_rn.shape[2:] → already static.
    #   Self.size is a safety net.
    scratch.refinenet2.size = size_l1

    # refinenet1: upsamples from size_l1 → size_l1*2
    #   Called WITHOUT any size= kwarg in BOTH 4-layer and 2-layer paths.
    #   Without this patch it falls through to scale_factor=2, which
    #   produces a dynamic Resize node. We replace it with a static size.
    scratch.refinenet1.size = (patch_h * 8, patch_w * 8)

    patched = ['refinenet4', 'refinenet3', 'refinenet2', 'refinenet1']
    print(f"patch_static_interpolation_sizes: baked static sizes into "
          f"{patched} for input ({input_h}, {input_w}).")
    print(f"  patch grid : ({patch_h}, {patch_w})")
    print(f"  refinenet4 : {size_l3}")
    print(f"  refinenet3 : {size_l2}")
    print(f"  refinenet2 : {size_l1}")
    print(f"  refinenet1 : {(patch_h * 8, patch_w * 8)}")

    return model


# ---------------------------------------------------------------------------
# Positional encoding cache pre-warm
# ---------------------------------------------------------------------------

def _prewarm_pos_enc_cache(model: nn.Module, input_size: tuple) -> None:
    """
    Pre-warm the DINOv2 positional encoding cache and register each cached
    tensor as a named nn.Module buffer so torch.export embeds it cleanly.

    Background
    ----------
    DinoVisionTransformer.interpolate_pos_encoding() caches its result in a
    plain Python dict (_pos_enc_cache) to avoid recomputing the bicubic
    interpolation on every call. When torch.export.export() traces the model,
    it runs one forward pass internally; this populates the cache. The tracer
    then encounters the dict assignment as an unregistered tensor attribute and
    emits:

        UserWarning: The tensor attribute self.model.core.pretrained
        ._pos_enc_cache[(504, 1008)] was assigned during export. Such
        attributes must be registered as buffers using the register_buffer API.

    More critically, the cached tensor ends up represented in the exported
    graph as a captured constant with no stable name, which can cause
    onnxsim and ORT to treat it inconsistently across runs.

    Fix
    ---
    We run a single no-grad forward pass here (before the tracer ever runs)
    to populate _pos_enc_cache. Then we iterate the cache and re-register
    every entry as a named buffer on the DinoVisionTransformer module using
    a deterministic name. The existing cache dict entries are replaced with
    direct references to those buffers so the forward path continues to work
    — but now when the tracer runs its own forward pass, the tensor is already
    a registered buffer and the warning is suppressed.

    The key encoding uses '_' instead of ',' and parentheses so the buffer
    name is a valid Python identifier (required by register_buffer).
    """
    if not (hasattr(model, 'core')
            and hasattr(model.core, 'pretrained')
            and hasattr(model.core.pretrained, '_pos_enc_cache')):
        return

    vit = model.core.pretrained
    _, _, h, w = input_size

    print(f"Pre-warming positional encoding cache for ({h}, {w})...")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w)
        try:
            model.core(dummy)
        except Exception:
            # If the core forward fails for any reason, the cache may still
            # have been partially populated — proceed regardless.
            pass

    # Promote every cached tensor to a registered buffer.
    registered = 0
    for (cw, ch), cached_tensor in list(vit._pos_enc_cache.items()):
        buf_name = f"_pos_enc_buf_{cw}_{ch}"
        # register_buffer makes it a proper module-owned tensor that the
        # tracer recognises as a weight/constant rather than a foreign assign.
        vit.register_buffer(buf_name, cached_tensor.detach(), persistent=False)
        # Keep the cache entry pointing at the now-registered buffer so
        # interpolate_pos_encoding() still finds and returns it correctly.
        vit._pos_enc_cache[(cw, ch)] = getattr(vit, buf_name)
        registered += 1

    print(f"  Registered {registered} positional encoding tensor(s) as module buffers.")


# ---------------------------------------------------------------------------
# Post-export ONNX passes
# ---------------------------------------------------------------------------

def run_onnxsim(onnx_path: str) -> str:
    """
    Run onnx-simplifier on the exported model.

    Constant-folds shape-dependent branches, removes dead nodes, and
    canonicalises Reshape/Gather patterns that confuse mobile runtimes.
    Writes a new file with '_sim' inserted before the extension.

    Requires: pip install onnxsim

    Returns the path to the simplified model, or the original path on failure.
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  onnxsim not installed — skipping (pip install onnxsim).")
        return onnx_path

    base, ext = os.path.splitext(onnx_path)
    out_path = f"{base}_sim{ext}"

    print(f"  Running onnxsim: {onnx_path} → {out_path}")
    try:
        model_onnx = onnx.load(onnx_path)
        model_simplified, check = simplify(model_onnx)
        if not check:
            print("  Warning: onnxsim validation failed — saving anyway.")
        onnx.save(model_simplified, out_path)

        original_size = os.path.getsize(onnx_path) / 1e6
        simplified_size = os.path.getsize(out_path) / 1e6
        print(f"  ✓ onnxsim complete: {original_size:.1f} MB → {simplified_size:.1f} MB  ({out_path})")
        return out_path
    except Exception as e:
        print(f"  ✗ onnxsim failed: {e} — using original.")
        return onnx_path


def run_ort_optimize(onnx_path: str) -> str:
    """
    Run OnnxRuntime's offline graph optimizer on the model.

    model_type='vit' activates ViT-specific transformer-library fusion passes:
    BiasGelu, LayerNormalization, SkipLayerNormalization, and (where the graph
    matches) MultiHeadAttention.  These fuse multi-node subgraphs into single
    custom ops that ORT executes with optimized kernels.

    CRITICAL — opt_level must be ORT_ENABLE_EXTENDED (2), NOT 99
    ---------------------------------------------------------------
    opt_level=99 enables ORT_ENABLE_ALL, which activates NchwcTransformer.
    That pass rewrites standard ONNX Conv nodes into com.microsoft.nchwc:Conv,
    a Microsoft-internal op that only exists in the x86/x64 desktop ORT runtime.
    The Android ORT runtime (arm64) has no such op, so the model fails to load:

        Fatal error: com.microsoft.nchwc:Conv(-1) is not a registered function/op

    ORT even warns about this during export:
        "Serializing optimized model with Graph Optimization level greater than
         ORT_ENABLE_EXTENDED and the NchwcTransformer enabled. The generated
         model may contain hardware specific optimizations, and should only be
         used in the same environment the model was optimized in."

    ORT_ENABLE_EXTENDED (2) applies all the transformer-library fusions
    (BiasGelu, LayerNorm, SkipLayerNorm, Attention) without NchwcTransformer.
    That is exactly what we want for cross-platform mobile deployment.

    The optimize_model() signature has changed across ORT versions:
      - 'optimization_level' existed in older versions, removed in ~1.17
      - 'opt_level' added in ~1.17
      - 'fusion_options' added later still (not present in all builds)
      - 'only_onnxruntime' also absent in some older versions

    We inspect the actual installed signature at runtime and pass only the
    kwargs that exist, so this function works across all ORT versions.

    Writes a new file with '_opt' inserted before the extension.

    Requires: pip install onnxruntime

    Returns the path to the optimized model, or the input path on failure.
    """
    try:
        from onnxruntime.transformers.optimizer import optimize_model
        import onnxruntime
        import inspect as _inspect
    except ImportError:
        print("  onnxruntime.transformers not installed — skipping "
              "(pip install onnxruntime).")
        return onnx_path

    base, ext = os.path.splitext(onnx_path)
    out_path = f"{base}_opt{ext}"

    print(f"  Running ORT optimizer: {onnx_path} -> {out_path}")
    print(f"  ORT version: {onnxruntime.__version__}")
    try:
        # Inspect the real signature so we never pass a kwarg that does not exist
        # in this particular ORT build.
        valid_params = set(_inspect.signature(optimize_model).parameters)

        kwargs = {
            "model_type": "vit",
            "num_heads": 0,    # 0 = auto-detect from graph
            "hidden_size": 0,  # 0 = auto-detect
            "use_gpu": False,  # CPU baseline — NNAPI/QNN EP loads on top
        }

        # ORT_ENABLE_EXTENDED = 2.
        # This is the highest level that is safe for cross-platform deployment.
        # Level 99 / ORT_ENABLE_ALL adds NchwcTransformer which inserts
        # com.microsoft.nchwc:Conv ops — these only run on x86/x64 and will
        # crash on Android with "not a registered function/op".
        ORT_ENABLE_EXTENDED = 2

        # opt_level / optimization_level — mutually exclusive across ORT versions
        if "opt_level" in valid_params:
            kwargs["opt_level"] = ORT_ENABLE_EXTENDED
        elif "optimization_level" in valid_params:
            kwargs["optimization_level"] = ORT_ENABLE_EXTENDED

        # only_onnxruntime — absent in some older builds
        if "only_onnxruntime" in valid_params:
            kwargs["only_onnxruntime"] = False

        # fusion_options — added in later ORT versions; import guard + param guard
        if "fusion_options" in valid_params:
            try:
                from onnxruntime.transformers.fusion_options import FusionOptions
                kwargs["fusion_options"] = FusionOptions("vit")
            except ImportError:
                pass  # older ORT without FusionOptions — skip silently

        log_kwargs = {k: v for k, v in kwargs.items() if k != "fusion_options"}
        print(f"  Calling optimize_model with: {log_kwargs}")

        optimized = optimize_model(onnx_path, **kwargs)
        optimized.save_model_to_file(out_path)

        original_size  = os.path.getsize(onnx_path) / 1e6
        optimized_size = os.path.getsize(out_path)  / 1e6

        # Report which operator patterns ORT actually fused.
        if hasattr(optimized, "get_fused_operator_statistics"):
            stats = optimized.get_fused_operator_statistics()
            if stats:
                print(f"  Fused operators: {stats}")

        print(f"  ORT optimize complete: {original_size:.1f} MB -> {optimized_size:.1f} MB  ({out_path})")
        return out_path
    except Exception as e:
        print(f"  ORT optimize failed: {e} — using input model.")
        import traceback
        traceback.print_exc()
        return onnx_path




# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_model_with_onnx(
    config_path,
    output_path,
    example_input_size=(1, 3, 504, 1008),
    use_fp16=False,
    merge_lora=True,
    fuse_layer_scale=True,
    fuse_bn=True,
    static_interp=True,
    onnx_sim=True,
    ort_optimize=True,
    num_encoder_layers=4,
):
    """
    Export model using ONNX for mobile deployment.

    Args:
        config_path:        Path to config YAML file
        output_path:        Path where exported model will be saved (.onnx)
        example_input_size: Tuple of (batch, channels, height, width) for export
        use_fp16:           FP32 I/O with internal FP16 weights
        merge_lora:         Merge LoRA A/B matrices into base QKV weights
        fuse_layer_scale:   Fuse LayerScale gamma into linear weights
        fuse_bn:            Fuse Conv2d+BatchNorm2d pairs into Conv2d
        static_interp:      Bake fixed upsample sizes into FeatureFusionBlocks
        onnx_sim:           Run onnx-simplifier after export
        ort_optimize:       Run ORT offline optimizer after export
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

    # --- LoRA merge ---
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

    # --- LayerScale fusion ---
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

    # --- BN fusion ---
    # Must run AFTER weights are loaded and model is in eval() mode.
    # BN running_mean/running_var are only meaningful post-training.
    # Must run BEFORE static_interp (which only touches FeatureFusionBlock,
    # but ordering is safer this way).
    if fuse_bn:
        print("Fusing Conv2d+BatchNorm2d pairs...")
        fuse_conv_bn(model)
    else:
        print("BN fusion skipped.")

    # --- Static interpolation sizes ---
    # Must run AFTER the model is fully configured (encoder layers set, etc.)
    # and BEFORE wrapping/tracing, so the tracer sees concrete size tuples.
    if static_interp:
        _, _, input_h, input_w = example_input_size
        print(f"Patching static interpolation sizes for input ({input_h}, {input_w})...")
        patch_static_interpolation_sizes(model, input_h, input_w)
    else:
        print("Static interpolation patch skipped.")

    # --- Pre-warm positional encoding cache and promote to registered buffer ---
    #
    # dinov2.py caches the bicubic-interpolated positional encoding in a plain
    # dict attribute (_pos_enc_cache) on first call. torch.export.export() runs
    # one forward pass to trace the graph, which populates that dict. The tracer
    # then sees the cached tensor as an unregistered attribute assignment and
    # emits a UserWarning, and — more importantly — may represent it as a
    # graph-captured constant with non-deterministic identity across export runs.
    #
    # Fix: run one warm-up forward pass NOW (before wrapping) so the cache is
    # populated, then re-register each cached tensor as a proper nn.Module buffer
    # under a stable name. The tracer will then find it as a named buffer (like
    # any other weight tensor) and embed it cleanly as an initializer in the
    # ONNX graph with no warning and no ambiguity.
    _prewarm_pos_enc_cache(model, example_input_size)

    # --- Choose wrapper ---
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
    print(f"  BN fused          : {fuse_bn}")
    print(f"  Static interp     : {static_interp}")
    print(f"  onnxsim           : {onnx_sim}")
    print(f"  ORT optimize      : {ort_optimize}")
    print(f"  Input size        : {example_input_size}")

    # --- ONNX export ---
    # Use the legacy TorchScript-based exporter (dynamo=False) rather than the
    # newer torch.export-based path. The dynamo exporter writes large models as
    # a split .onnx + .onnx.data pair (external tensors), producing a misleading
    # 1.1 MB base file — weights are stored separately and only inlined by
    # onnxsim in the next step. The legacy exporter always writes a single
    # self-contained file with all initializers embedded, which is what mobile
    # tools (onnxsim, ORT, NNAPI converter) expect as direct input.
    print(f"Exporting model with input size {example_input_size}...")
    try:
        print("Running torch.onnx.export()...")
        torch.onnx.export(
            wrapped_model,
            sample_inputs,
            output_path,
            input_names=["input"],
            output_names=["depth"],
            opset_version=18,   # 18+ required; Resize op has no downgrade adapter to 17
            dynamic_axes=None,  # fixed shape — matches your CoreML approach
            dynamo=False,       # legacy exporter: single self-contained .onnx,
                                # no external .onnx.data split
        )
        print(f"✓ Base export complete: {output_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # --- Post-export passes ---
    # Each pass reads from the previous output and writes a new suffixed file,
    # so you keep every intermediate artifact for comparison/debugging.
    current_path = output_path

    if onnx_sim:
        print("Running onnx-simplifier...")
        current_path = run_onnxsim(current_path)

    if ort_optimize:
        print("Running ORT graph optimizer...")
        current_path = run_ort_optimize(current_path)

    print(f"\n✓ Final model: {current_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument('--encoder-layers', type=int, default=4, choices=[2, 4],
                        help='Encoder layers at inference (2 or 4). Default: 4.')

    # Flags that were previously always-on are now opt-out for full control
    parser.add_argument('--no-merge-lora', action='store_true',
                        help='Skip LoRA weight merging.')
    parser.add_argument('--no-fuse-layer-scale', action='store_true',
                        help='Skip LayerScale fusion into linear weights.')
    parser.add_argument('--no-fuse-bn', action='store_true',
                        help='Skip Conv2d+BatchNorm2d fusion.')
    parser.add_argument('--no-static-interp', action='store_true',
                        help='Skip static interpolation size patching.')
    parser.add_argument('--no-onnx-sim', action='store_true',
                        help='Skip onnx-simplifier post-processing.')
    parser.add_argument('--no-ort-optimize', action='store_true',
                        help='Skip ORT offline graph optimization.')

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
        fuse_bn=not args.no_fuse_bn,
        static_interp=not args.no_static_interp,
        onnx_sim=not args.no_onnx_sim,
        ort_optimize=not args.no_ort_optimize,
        num_encoder_layers=args.encoder_layers,
    )

    if success:
        print("Export completed successfully!")
