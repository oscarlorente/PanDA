# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def fuse_layer_scale_into_linear(model: nn.Module) -> int:
    """
    Opt 4: Fuse LayerScale gamma permanently into the preceding linear layer.

    Each transformer block applies LayerScale after attention and after MLP:
        out = x + ls1(attn(norm1(x)))    # ls1 = element-wise * gamma1
        out = x + ls2(mlp(norm2(x)))     # ls2 = element-wise * gamma2

    LayerScale is mathematically equivalent to scaling the output rows of the
    preceding linear layer (self.attn.proj and self.mlp.fc2) by gamma.
    After fusion:
        proj.weight[i, :] *= gamma1[i]   for each output feature i
        proj.bias[i]      *= gamma1[i]
        fc2.weight[i, :]  *= gamma2[i]
        fc2.bias[i]       *= gamma2[i]

    The fused model is exactly numerically equivalent to the original at
    float32 precision.  This eliminates 2 element-wise multiplications per
    block — 24 total for ViT-S (12 blocks × 2).

    Call AFTER loading weights and BEFORE torch.export.export().
    The LayerScale modules are replaced with nn.Identity() after fusion.

    Args:
        model: a DinoVisionTransformer (da_model.pretrained)

    Returns:
        number of LayerScale modules fused
    """
    fused = 0
    for blk in model.blocks:
        # --- Fuse ls1 into attn.proj ---
        if isinstance(blk.ls1, LayerScale) and not isinstance(blk.ls1, nn.Identity):
            gamma1 = blk.ls1.gamma.data  # shape [embed_dim]
            proj = blk.attn.proj         # nn.Linear(embed_dim, embed_dim)
            proj.weight.data *= gamma1.unsqueeze(1)   # scale each output row
            if proj.bias is not None:
                proj.bias.data *= gamma1
            blk.ls1 = nn.Identity()
            fused += 1

        # --- Fuse ls2 into mlp.fc2 ---
        if isinstance(blk.ls2, LayerScale) and not isinstance(blk.ls2, nn.Identity):
            gamma2 = blk.ls2.gamma.data  # shape [embed_dim]
            fc2 = blk.mlp.fc2            # nn.Linear(hidden_dim, embed_dim)
            fc2.weight.data *= gamma2.unsqueeze(1)    # scale each output row
            if fc2.bias is not None:
                fc2.bias.data *= gamma2
            blk.ls2 = nn.Identity()
            fused += 1

    print(f"fuse_layer_scale_into_linear: fused {fused} LayerScale modules.")
    return fused
