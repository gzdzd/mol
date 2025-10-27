# models/blending_module.py
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .atom_group_moe import GroupedAtomMoE


class BlendingModule(nn.Module):
    """
    Adapter module that fuses multi-encoder graph features into a single sequence
    using the grouped atom-level MoE (2D/3D/mix) while keeping the original
    project-facing interface.

    Expected by MolLLaMAEncoder:
      - __init__(hidden_dim, num_heads, num_layers, dims)
      - forward(batch_nodes: Dict[str, Tensor], batch_masks: Dict[str, Tensor])
        -> returns (batch_node, batch_mask, stats)

    Args:
        hidden_dim (int): target fused hidden size (df)
        num_heads (int): kept for interface compatibility (unused here)
        num_layers (int): kept for interface compatibility (unused here)
        dims (Dict[str, int]): encoder name -> feature dim mapping
                              e.g., {"moleculestm": d2d, "unimol": d3d}
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dims: Dict[str, int],
    ):
        super().__init__()
        if not isinstance(dims, dict) or len(dims) < 2:
            raise ValueError("BlendingModule requires at least two encoders in dims.")

        # Prefer conventional names; otherwise fall back to the first two keys deterministically
        if "moleculestm" in dims and "unimol" in dims:
            self.key_2d = "moleculestm"
            self.key_3d = "unimol"
        else:
            keys = sorted(list(dims.keys()))
            self.key_2d, self.key_3d = keys[0], keys[1]

        d2d = int(dims[self.key_2d])
        d3d = int(dims[self.key_3d])
        self.hidden_dim = int(hidden_dim)
        self.dims = dims

        # Grouped MoE fuser to combine per-atom 2D/3D features into df
        self.fuser = GroupedAtomMoE(
            d2d=d2d,
            d3d=d3d,
            df=self.hidden_dim,
            n_experts=4,
            topk=1,  # default to hard top-1 + capacity for efficiency; can be tuned
            gate_temp=1.2,
            gate_noise=0.3,
            dropout=0.1,
            capacity_factor=1.25,
            use_gumbel=False,
            use_hard_capacity=True,
        )

    def forward(
        self,
        batch_nodes: Dict[str, torch.Tensor],
        batch_masks: Dict[str, torch.Tensor],
        return_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Inputs:
          batch_nodes: dict of encoder_name -> [B, N, d]
          batch_masks: dict of encoder_name -> [B, N] (bool)
          return_stats: whether to return routing stats from fuser
        Returns:
          fused_nodes: [B, N, hidden_dim]
          fused_mask:  [B, N] (bool)
          stats: dict (or None)
        """
        if self.key_2d not in batch_nodes or self.key_3d not in batch_nodes:
            raise KeyError(
                f"Expected encoders '{self.key_2d}' and '{self.key_3d}' in batch_nodes. Got {list(batch_nodes.keys())}."
            )
        h2d = batch_nodes[self.key_2d]
        h3d = batch_nodes[self.key_3d]

        # 防御性对齐：若两路序列长度不一致，统一裁剪到最短长度
        L2 = h2d.size(1)
        L3 = h3d.size(1)
        L = min(L2, L3)
        if L2 != L3:
            h2d = h2d[:, :L, :]
            h3d = h3d[:, :L, :]

        mask2d = batch_masks.get(self.key_2d, None)
        mask3d = batch_masks.get(self.key_3d, None)
        if mask2d is not None and mask2d.size(1) != L:
            mask2d = mask2d[:, :L]
        if mask3d is not None and mask3d.size(1) != L:
            mask3d = mask3d[:, :L]

        if mask2d is not None and mask3d is not None:
            fused_mask = (mask2d.bool() & mask3d.bool())
        elif mask2d is not None:
            fused_mask = mask2d.bool()
        elif mask3d is not None:
            fused_mask = mask3d.bool()
        else:
            fused_mask = None

        out, stats = self.fuser(h2d, h3d, mask=fused_mask, return_stats=return_stats)
        # Make sure mask is a Tensor of bools even when None
        if fused_mask is None:
            fused_mask = torch.ones(out.shape[:2], dtype=torch.bool, device=out.device)

        return out, fused_mask, stats if return_stats else None
