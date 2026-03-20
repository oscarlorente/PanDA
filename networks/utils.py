import math
import copy
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import save_file
from safetensors import safe_open
from torch.nn.parameter import Parameter

from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2_metric.depth_anything_v2.dinov2_layers.layer_scale import fuse_layer_scale_into_linear


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


def merge_lora_weights(da_model):
    """
    Merge LoRA A and B matrices permanently into the base QKV weights.

    During normal inference, each _LoRA_qkv module runs the base linear plus
    two rank-4 projections (A_q, B_q) and (A_v, B_v) on every token at every
    layer. With ViT-S (12 blocks, rank 4), that is 48 extra linear ops per
    forward pass — all of which can be eliminated because:

        W_q_merged = W_q + B_q @ A_q
        W_v_merged = W_v + B_v @ A_v

    The merged model is mathematically identical to the original but has
    simpler graph structure, fewer nodes for XNNPACK to deal with, and
    eliminates all LoRA dispatch overhead at runtime.

    Call this function AFTER loading the trained weights and BEFORE
    torch.export.export().  After merging, the model no longer contains any
    _LoRA_qkv modules — each block's attn.qkv is a plain nn.Linear.

    Args:
        da_model: a DepthAnythingV2 instance whose pretrained blocks may
                  contain _LoRA_qkv attention modules.

    Returns:
        da_model (in-place modified)
    """
    merged_count = 0
    for blk in da_model.pretrained.blocks:
        if not isinstance(blk.attn.qkv, _LoRA_qkv):
            continue

        lora_qkv = blk.attn.qkv
        base_linear = lora_qkv.qkv          # the original nn.Linear
        dim = lora_qkv.dim                   # embed_dim (384 for ViT-S)

        # Work in float32 for numerical accuracy regardless of storage dtype
        W = base_linear.weight.data.float()  # shape: [3*dim, dim]

        # Merge Q branch:  delta_q = B_q @ A_q  → shape [dim, dim]
        A_q = lora_qkv.linear_a_q.weight.data.float()  # [r, dim]
        B_q = lora_qkv.linear_b_q.weight.data.float()  # [dim, r]
        W[:dim] += B_q @ A_q

        # Merge V branch:  delta_v = B_v @ A_v  → shape [dim, dim]
        A_v = lora_qkv.linear_a_v.weight.data.float()  # [r, dim]
        B_v = lora_qkv.linear_b_v.weight.data.float()  # [dim, r]
        W[-dim:] += B_v @ A_v

        # Write merged weights back in the original dtype
        base_linear.weight.data = W.to(base_linear.weight.dtype)

        # Replace _LoRA_qkv with the plain base linear — LoRA is gone
        blk.attn.qkv = base_linear
        merged_count += 1

    print(f"merge_lora_weights: merged {merged_count} LoRA blocks into base weights.")
    return da_model


class LoRA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """
        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        load both lora and fc parameters.
        """
        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class LoRA_Depth_Anything_v2(LoRA):
    """Applies low-rank adaptation to a Depth Anything model's image encoder.

    Args:
        da_model: a DepthAnythingV2 instance
        r: rank of LoRA
        lora_layer: which layers to apply LoRA (default: all)
    """

    def __init__(self, da_model: DepthAnythingV2, r: int, lora_layer=None):
        super(LoRA_Depth_Anything_v2, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(da_model.pretrained.blocks)))

        self.w_As = []
        self.w_Bs = []

        # Freeze encoder
        for param in da_model.pretrained.parameters():
            param.requires_grad = False

        # Inject LoRA into each attention block
        for t_layer_i, blk in enumerate(da_model.pretrained.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

        self.lora_vit = da_model
