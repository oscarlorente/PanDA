from __future__ import absolute_import, division, print_function
"""
TFLite Model Export Script for Mobile Deployment (Android GPU)

Export pipeline:
  PyTorch → ONNX → onnxsim → surgery → TFLite (FP16, GPU delegate ready)

Pipeline design
---------------
Two problems must be solved in sequence:

  PROBLEM 1 — node_view_N (Reshape) crash without onnxsim:
    ViT patch-embedding Reshape nodes have their shape arg as a bare constant
    initializer. onnx2tf tries to trace it as a Keras layer output → crash.
    FIX: run onnxsim first to fold constants into the graph.

  PROBLEM 2 — node_add_N spatial mismatches:
    The DPT decoder adds feature maps from different transformer layers at
    different resolutions (one per scale level). onnxruntime cannot propagate
    shapes through the custom attention ops, so infer_shapes() leaves the Add
    inputs with unknown shapes. The auto-JSON fix is a no-op identity perm.
    FIX: scan the post-onnxsim ONNX graph, find *all* Add nodes whose inputs
    have mismatched spatial dims using every available shape source (value_info,
    initializers, graph I/O), and insert Resize nodes before the smaller input.
    Where shapes are still unknown after all sources are exhausted, patch by
    the stable post-onnxsim node name with hardcoded scale factors derived from
    the observed NHWC error shapes reported by onnx2tf.

  PROBLEM 3 — ConvTranspose renaming:
    onnxsim folds ConvTranspose → Conv internally but leaves op attributes that
    make onnx2tf use its ConvTranspose code path (wrong permutation → crash).
    FIX: re-patch ConvTranspose → Conv after onnxsim runs.

Requirements:
  pip install onnx onnx2tf onnxsim

Android runtime:
  implementation 'org.tensorflow:tensorflow-lite:+'
  implementation 'org.tensorflow:tensorflow-lite-gpu:+'
  implementation 'org.tensorflow:tensorflow-lite-gpu-api:+'
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
        if isinstance(output, dict):
            return output['pred_depth']
        return output


# ---------------------------------------------------------------------------
# Graph surgery helpers
# ---------------------------------------------------------------------------

def fix_convtranspose_weights(model):
    """
    In-place: replace every ConvTranspose node with an equivalent Conv node.
    Must run AFTER onnxsim (which renames them but leaves wrong op attributes).
    """
    import onnx
    import numpy as np
    from onnx import numpy_helper, helper

    init_index = {init.name: i for i, init in enumerate(model.graph.initializer)}
    nodes_to_replace = []
    patched = 0

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type != "ConvTranspose":
            continue
        if len(node.input) < 2 or not node.input[1]:
            continue
        weight_name = node.input[1]
        if weight_name not in init_index:
            continue
        idx = init_index[weight_name]
        w = numpy_helper.to_array(model.graph.initializer[idx])
        if w.ndim != 4:
            continue
        w_fixed = w.transpose(1, 0, 2, 3).copy()
        model.graph.initializer[idx].CopyFrom(
            numpy_helper.from_array(w_fixed, name=weight_name))
        conv_attrs = {}
        for attr in node.attribute:
            if attr.name in ("dilations", "group", "kernel_shape", "pads", "strides"):
                conv_attrs[attr.name] = onnx.helper.get_attribute_value(attr)
        new_node = helper.make_node(
            op_type="Conv",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            **conv_attrs,
        )
        nodes_to_replace.append((node_idx, new_node))
        patched += 1
        print(f"    ✓  '{node.name}': ConvTranspose→Conv, weight {list(w.shape)}")

    for node_idx, new_node in reversed(nodes_to_replace):
        del model.graph.node[node_idx]
        model.graph.node.insert(node_idx, new_node)

    return patched


def _build_shape_map(model):
    """
    Build a tensor-name → shape list mapping from all available sources.

    Sources (in priority order):
      1. Initializers — always fully known (constant weight tensors)
      2. value_info   — intermediate activations with shape annotations
      3. Graph inputs / outputs

    After onnxsim, value_info is partially stripped for tensors that flow
    through custom attention ops onnxruntime cannot infer. We collect what
    we can and return None for unknowns.
    """
    shape_map = {}

    # 1. Initializers — dims are always concrete
    for init in model.graph.initializer:
        shape_map[init.name] = list(init.dims)

    # 2 & 3. Typed tensor descriptors
    for vi in (list(model.graph.value_info)
               + list(model.graph.input)
               + list(model.graph.output)):
        if vi.name in shape_map:
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else None)
        shape_map[vi.name] = dims

    return shape_map


def fix_all_spatial_add_mismatches(model):
    """
    In-place: patch every Add node whose inputs have mismatched spatial dims.

    Two-tier strategy
    -----------------
    Tier 1 — KNOWN_ADD_FIXES (always wins, checked first):
        DPT cross-scale Add nodes have stable post-onnxsim names. Their input
        shapes cannot be reliably resolved because onnxsim sometimes propagates
        incorrect shapes through the custom attention ops (making shape_a ==
        shape_b even when they are not). We bypass shape detection entirely for
        these nodes and apply hardcoded scale factors derived from the NHWC
        shapes onnx2tf reports in its error messages.

        IMPORTANT: if onnx2tf reports a new node name (e.g. node_add_60), add
        it to this table using the H ratio from the error shapes as the scale.

    Tier 2 — shape-based detection (for all other Add nodes):
        If both inputs have fully-known 4-D NCHW shapes and the spatial dims
        differ, insert a Resize using exact target sizes.
    """
    import numpy as np
    from onnx import numpy_helper, helper

    # Tier-1 table.  Key: post-onnxsim node name (stable).
    # Value: (small_input_idx, h_scale, w_scale) — NCHW scale factors.
    # Derived from NHWC shapes in onnx2tf error output:
    #   node_add_54: x=[1,72,144,64]  y=[1,18,36,64]  → y is small, 4×
    #   node_add_57: x=[1,144,288,64] y=[1,9,18,64]   → y is small, 16×
    KNOWN_ADD_FIXES = {
        "node_add_54": (1, 4.0,  4.0),
        "node_add_57": (1, 16.0, 16.0),
    }

    shape_map = _build_shape_map(model)
    resize_ops = []  # list of (add_node_idx, resize_node)
    counter = 0

    # Print all Add names so we can verify the table if it misses a node
    add_names = [n.name for n in model.graph.node if n.op_type == "Add"]
    print(f"    [debug] Add nodes post-onnxsim: {add_names}")

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type != "Add" or len(node.input) != 2:
            continue

        # --- Tier 1: hardcoded table always wins ---
        if node.name in KNOWN_ADD_FIXES:
            small_idx, h_scale, w_scale = KNOWN_ADD_FIXES[node.name]
            small_inp = node.input[small_idx]
            resized_name = small_inp + f"__up{counter}"
            node.input[small_idx] = resized_name

            scales  = np.array([1.0, 1.0, h_scale, w_scale], dtype=np.float32)
            s_name  = f"__resize_scales_{counter}"
            roi_name = f"__resize_roi_{counter}"
            model.graph.initializer.append(
                numpy_helper.from_array(scales, name=s_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array([], dtype=np.float32),
                                        name=roi_name))
            resize_node = helper.make_node(
                op_type="Resize",
                inputs=[small_inp, roi_name, s_name],
                outputs=[resized_name],
                name=f"__resize_fix_{counter}",
                coordinate_transformation_mode="asymmetric",
                mode="nearest",
                nearest_mode="floor",
            )
            print(f"    ✓  '{node.name}': Resize input[{small_idx}] ×{h_scale:.0f} "
                  f"(hardcoded)")
            resize_ops.append((node_idx, resize_node))
            counter += 1
            continue

        # --- Tier 2: shape-based detection ---
        inp_a, inp_b = node.input[0], node.input[1]
        shape_a = shape_map.get(inp_a)
        shape_b = shape_map.get(inp_b)
        a_ok = (shape_a is not None and len(shape_a) == 4
                and None not in shape_a)
        b_ok = (shape_b is not None and len(shape_b) == 4
                and None not in shape_b)
        if not (a_ok and b_ok):
            continue

        ha, wa = shape_a[2], shape_a[3]
        hb, wb = shape_b[2], shape_b[3]
        if ha == hb and wa == wb:
            continue

        if ha * wa >= hb * wb:
            small_inp, large_shape = inp_b, shape_a
            resized_name = inp_b + f"__up{counter}"
            node.input[1] = resized_name
        else:
            small_inp, large_shape = inp_a, shape_b
            resized_name = inp_a + f"__up{counter}"
            node.input[0] = resized_name

        target_size = np.array(large_shape, dtype=np.int64)
        sz_name  = f"__resize_size_{counter}"
        roi_name = f"__resize_roi_{counter}"
        model.graph.initializer.append(
            numpy_helper.from_array(target_size, name=sz_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([], dtype=np.float32),
                                    name=roi_name))
        resize_node = helper.make_node(
            op_type="Resize",
            inputs=[small_inp, roi_name, "", sz_name],
            outputs=[resized_name],
            name=f"__resize_fix_{counter}",
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
        )
        print(f"    ✓  '{node.name}': Resize {shape_map.get(small_inp,'?')} "
              f"→ {large_shape} (shape-based)")
        resize_ops.append((node_idx, resize_node))
        counter += 1

    # Insert each Resize immediately before its Add (reverse order preserves
    # indices for earlier nodes)
    for add_node_idx, resize_node in reversed(resize_ops):
        model.graph.node.insert(add_node_idx, resize_node)

    return counter


def run_onnxsim(onnx_model):
    """Run onnxsim and return (model, success_bool)."""
    try:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if ok:
            return simplified, True
        print("    ⚠  onnxsim could not simplify — using original.")
        return onnx_model, False
    except ImportError:
        print("    ⚠  onnxsim not installed (pip install onnxsim) — skipping.")
        return onnx_model, False
    except Exception as e:
        print(f"    ⚠  onnxsim failed ({e}) — using original.")
        return onnx_model, False


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_model_with_tflite(
    config_path,
    output_path,
    example_input_size=(1, 3, 504, 1008),
    use_fp16=True,
):
    import onnx

    # -- 1. Load model --------------------------------------------------------
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading model weights...")
    model_dict = torch.load(config["model_path"], map_location='cpu')
    model = make(config['model'])

    if any(key.startswith('module') for key in model_dict.keys()):
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

    model_state_dict = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in model_dict.items() if k in model_state_dict}
    )
    model.eval()

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    print("Model loaded successfully.")

    base_path = os.path.splitext(output_path)[0]

    # -- 2. Export to ONNX ----------------------------------------------------
    onnx_path = base_path + "_tmp.onnx"
    sample_input = (torch.randn(example_input_size),)

    print(f"\nExporting to ONNX: {onnx_path} ...")
    try:
        torch.onnx.export(
            wrapped_model,
            sample_input,
            onnx_path,
            input_names=["input"],
            output_names=["depth"],
            opset_version=18,
            dynamic_axes=None,
        )
        print(f"  ✓ ONNX export done.")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # -- 3. Graph surgery pipeline -------------------------------------------
    #
    # Step order matters:
    #   (a) onnxsim   — fold Reshape constant inputs (fixes node_view crash)
    #   (b) ConvTranspose → Conv  (onnxsim breaks these, must re-patch after)
    #   (c) All spatial Add mismatches  (shape-based scan + hardcoded fallback)
    #   (d) Validate + save once

    print("\nApplying ONNX graph surgery...")
    onnx_model = onnx.load(onnx_path)

    print("  [3a] Running onnxsim (Reshape constant folding)...")
    onnx_model, sim_ok = run_onnxsim(onnx_model)
    print(f"       {'✓ Simplified.' if sim_ok else 'ℹ Skipped.'}")

    print("  [3b] Fixing ConvTranspose → Conv...")
    n_ct = fix_convtranspose_weights(onnx_model)
    print(f"       {'✓ Replaced ' + str(n_ct) + ' node(s).' if n_ct else 'ℹ None found.'}")

    print("  [3c] Fixing spatial Add mismatches (all nodes)...")
    n_add = fix_all_spatial_add_mismatches(onnx_model)
    print(f"       {'✓ Fixed ' + str(n_add) + ' node(s).' if n_add else 'ℹ None found.'}")

    onnx_patched_path = base_path + "_tmp_patched.onnx"
    try:
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX graph check passed.")
    except Exception as e:
        print(f"  ⚠ ONNX checker warning (may be benign): {e}")
    onnx.save(onnx_model, onnx_patched_path)
    print(f"  ✓ Patched ONNX saved: {onnx_patched_path}")

    # -- 4. Convert ONNX → TFLite via onnx2tf --------------------------------
    print(f"\nConverting ONNX → TFLite via onnx2tf ...")
    print(f"  Input  : {onnx_patched_path}")
    print(f"  Output : {output_path}")
    try:
        import onnx2tf

        output_dir = os.path.dirname(output_path) or "."
        auto_json_path = os.path.splitext(onnx_patched_path)[0] + "_auto.json"

        # Remove any stale auto JSON from a previous run. If it exists on
        # attempt 1, onnx2tf picks it up immediately — but it was generated
        # against a differently-patched graph and will be wrong.
        if os.path.exists(auto_json_path):
            os.remove(auto_json_path)
            print(f"  Removed stale auto JSON: {auto_json_path}")

        MAX_ATTEMPTS = 5
        last_error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            param_json = auto_json_path if os.path.exists(auto_json_path) else None
            print(f"  [attempt {attempt}] "
                  f"{'Using param_replacement: ' + param_json if param_json else 'No param_replacement JSON yet.'}")
            try:
                onnx2tf.convert(
                    input_onnx_file_path=onnx_patched_path,
                    output_folder_path=output_dir,
                    param_replacement_file=param_json,
                    output_integer_quantized_tflite=False,
                    quant_type=None,
                    not_use_onnxsim=True,
                    auto_generate_json_on_error=True,
                    verbosity="warn",
                )
                print(f"  ✓ Conversion succeeded on attempt {attempt}.")
                last_error = None
                break
            except Exception as e:
                last_error = e
                if os.path.exists(auto_json_path):
                    print(f"  ⚠ Attempt {attempt} failed — auto JSON updated, retrying...")
                else:
                    print(f"  ⚠ Attempt {attempt} failed — no auto JSON produced.")
                    break

        if last_error is not None:
            raise last_error

        # onnx2tf writes the .tflite inside output_dir but the exact subpath
        # varies by version. Search recursively rather than guessing.
        src = None
        for root, _, files in os.walk(output_dir):
            for fname in sorted(files):  # sorted for determinism
                if fname.endswith(".tflite"):
                    src = os.path.join(root, fname)
                    break
            if src:
                break
        if src is None:
            print("  ✗ Could not find generated .tflite file.")
            print(f"    Searched under: {output_dir}")
            # Print tree to help diagnose
            for root, dirs, files in os.walk(output_dir):
                print(f"    {root}: {files}")
            return False
        print(f"  ✓ Found .tflite at: {src}")
        if src != output_path:
            import shutil
            shutil.move(src, output_path)
        print(f"  ✓ TFLite model saved: {output_path}")

    except ImportError:
        print("  ✗ onnx2tf not installed. Install with: pip install onnx2tf")
        return False
    except Exception as e:
        print(f"  ✗ onnx2tf conversion failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # -- 5. FP16 weight quantisation ------------------------------------------
    if use_fp16:
        print("\nApplying FP16 weight optimisation...")
        try:
            import tensorflow as tf

            # Find the saved_model directory onnx2tf wrote
            saved_model_dir = None
            for root, dirs, files in os.walk(output_dir):
                if "saved_model.pb" in files:
                    saved_model_dir = root
                    break

            if saved_model_dir is None:
                print("  ⚠ saved_model directory not found — skipping FP16 pass.")
            else:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                tflite_fp16 = converter.convert()
                fp16_path = base_path + "_fp16.tflite"
                with open(fp16_path, "wb") as f:
                    f.write(tflite_fp16)
                print(f"  ✓ FP16 TFLite model saved: {fp16_path}")
        except ImportError:
            print("  ⚠ TensorFlow not installed — skipping FP16 pass.")
        except Exception as e:
            print(f"  ⚠ FP16 pass failed ({e}) — FP32 model is still usable.")

    # -- 6. Cleanup -----------------------------------------------------------
    for tmp in [base_path + "_tmp.onnx",
                base_path + "_tmp_patched.onnx",
                base_path + "_tmp_patched_auto.json"]:
        if os.path.exists(tmp):
            os.remove(tmp)
            print(f"  Removed temp file: {tmp}")

    # -- 7. Summary -----------------------------------------------------------
    print("\n================================================")
    print("Export complete!")
    print(f"  Model input size : {example_input_size}")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  FP32 TFLite size : {size_mb:.1f} MB  →  {output_path}")
    fp16_path = base_path + "_fp16.tflite"
    if os.path.exists(fp16_path):
        size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
        print(f"  FP16 TFLite size : {size_mb:.1f} MB  →  {fp16_path}")
    print("================================================")
    print("\nAndroid integration (build.gradle):")
    print("  implementation 'org.tensorflow:tensorflow-lite:+'")
    print("  implementation 'org.tensorflow:tensorflow-lite-gpu:+'")
    print("  implementation 'org.tensorflow:tensorflow-lite-gpu-api:+'")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export model to TFLite with GPU-delegate support'
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str,
                        default='./checkpoints/model_mobile.tflite')
    parser.add_argument('--height', type=int, default=504,
                        help='Input height (width = 2×height, must be multiple of 14)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-fp16', action='store_true',
                        help='Skip FP16 weight quantisation pass')

    args = parser.parse_args()
    erp_height = args.height
    erp_width = 2 * erp_height
    input_size = (args.batch_size, 3, erp_height, erp_width)

    print("TFLite Model Export for Android GPU Delegate")
    print("============================================")
    success = export_model_with_tflite(
        config_path=args.config,
        output_path=args.output,
        example_input_size=input_size,
        use_fp16=not args.no_fp16,
    )
    if success:
        print("\nExport completed successfully!")
    else:
        print("\nExport failed. See error messages above.")
        exit(1)
