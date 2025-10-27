# -*- coding: utf-8 -*-
"""
Quick smoke tests for GroupedAtomMoE.
- Validates shapes, masking behavior, and stats for both hard (Top-1+capacity) and soft (Top-k) routing.
- Synthetic inputs; no dependency on full pipeline.
"""
import os
import math
import torch
import torch.nn as nn

from models.blending_module.atom_group_moe import GroupedAtomMoE


def make_batch(B=2, N=8, d2d=64, d3d=80, device="cpu"):
    torch.manual_seed(42)
    h2d = torch.randn(B, N, d2d, device=device)
    h3d = torch.randn(B, N, d3d, device=device)
    # Mask: first half valid, second half invalid for each batch
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    valid_len = N // 2
    mask[:, :valid_len] = True
    return h2d, h3d, mask


def run_hard_routing(device="cpu"):
    print("[Hard routing] Top-1 + capacity")
    B, N, d2d, d3d, df, K = 3, 10, 48, 96, 128, 3
    h2d, h3d, mask = make_batch(B, N, d2d, d3d, device=device)

    moe = GroupedAtomMoE(
        d2d=d2d,
        d3d=d3d,
        df=df,
        n_experts=K,
        topk=1,  # triggers hard routing when use_hard_capacity=True
        gate_temp=1.1,
        gate_noise=0.0,
        dropout=0.05,
        capacity_factor=1.25,
        use_gumbel=False,
        use_hard_capacity=True,
    ).to(device)

    out, stats = moe(h2d, h3d, mask=mask, return_stats=True)

    # Shape checks
    assert out.shape == (B, N, df), f"Unexpected out shape: {out.shape}"

    # Masked positions should be zeroed
    masked_out = out[~mask]
    assert torch.allclose(masked_out, torch.zeros_like(masked_out), atol=1e-6), "Masked positions not zeroed"

    # Stats checks
    assert "group_importance" in stats and stats["group_importance"].numel() == 3, "Missing group_importance"
    gi = stats["group_importance"]
    assert torch.all(gi >= 0) and torch.all(gi <= 1), "group_importance out of range"

    # Expert importance per group should sum to ~1 across K
    for key in ("expert_importance_2d", "expert_importance_3d", "expert_importance_mix"):
        imp = stats[key]
        s = imp.sum().item()
        assert imp.numel() == K and abs(s - 1.0) < 1e-3, f"{key} does not sum to 1 (got {s})"

    # Expert load exists and is non-negative
    for key in ("expert_load_2d", "expert_load_3d", "expert_load_mix"):
        load = stats[key]
        assert load.numel() == K and torch.all(load >= 0), f"{key} invalid"

    print("  Output shape:", tuple(out.shape))
    print("  group_importance:", stats["group_importance"].tolist())
    print("  expert_importance_2d:", stats["expert_importance_2d"].tolist())
    print("  expert_importance_3d:", stats["expert_importance_3d"].tolist())
    print("  expert_importance_mix:", stats["expert_importance_mix"].tolist())
    print("  expert_load_2d:", stats["expert_load_2d"].tolist())
    print("  expert_load_3d:", stats["expert_load_3d"].tolist())
    print("  expert_load_mix:", stats["expert_load_mix"].tolist())



def run_soft_routing(device="cpu"):
    print("[Soft routing] Top-k weighted (k=2)")
    B, N, d2d, d3d, df, K = 2, 12, 32, 72, 96, 4
    h2d, h3d, mask = make_batch(B, N, d2d, d3d, device=device)

    moe = GroupedAtomMoE(
        d2d=d2d,
        d3d=d3d,
        df=df,
        n_experts=K,
        topk=2,  # soft routing path
        gate_temp=1.2,
        gate_noise=0.0,
        dropout=0.1,
        capacity_factor=1.25,
        use_gumbel=False,
        use_hard_capacity=False,
    ).to(device)

    out, stats = moe(h2d, h3d, mask=mask, return_stats=True)

    # Shape checks
    assert out.shape == (B, N, df), f"Unexpected out shape: {out.shape}"

    # Masked positions should be zeroed
    masked_out = out[~mask]
    assert torch.allclose(masked_out, torch.zeros_like(masked_out), atol=1e-6), "Masked positions not zeroed"

    # Stats checks
    assert "group_importance" in stats and stats["group_importance"].numel() == 3, "Missing group_importance"
    gi = stats["group_importance"]
    assert torch.all(gi >= 0) and torch.all(gi <= 1), "group_importance out of range"

    # Expert importance per group should sum to ~1 across K (Top-k normalized)
    for key in ("expert_importance_2d", "expert_importance_3d", "expert_importance_mix"):
        imp = stats[key]
        s = imp.sum().item()
        assert imp.numel() == K and abs(s - 1.0) < 1e-3, f"{key} does not sum to 1 (got {s})"

    print("  Output shape:", tuple(out.shape))
    print("  group_importance:", stats["group_importance"].tolist())
    print("  expert_importance_2d:", stats["expert_importance_2d"].tolist())
    print("  expert_importance_3d:", stats["expert_importance_3d"].tolist())
    print("  expert_importance_mix:", stats["expert_importance_mix"].tolist())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    run_hard_routing(device=device)
    run_soft_routing(device=device)
    print("All tests passed.")