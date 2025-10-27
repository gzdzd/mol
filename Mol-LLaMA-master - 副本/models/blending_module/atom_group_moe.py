# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExpertFFN(nn.Module):
    def __init__(self, df: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = df * expansion
        self.fc1 = nn.Linear(df, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, df)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class GroupedAtomMoE(nn.Module):
    """
    原子级分组MoE：三组（2D/3D/混合），每组内包含若干专家。
    - 组别门控：基于 2D/3D/融合 三路输入产生组别概率 p2d/p3d/pmix
    - 组内路由：每组独立的专家门控与 FFN 专家；支持 Top-1 + 容量，或 Top-k 软加权

    Inputs:
      h2d: [B, N, d2d]
      h3d: [B, N, d3d]
      mask: [B, N]  True 表示有效原子位置
    Returns:
      out: [B, N, df]
      stats: dict（可选）包含组别与专家的路由统计
    """
    def __init__(
        self,
        d2d: int,
        d3d: int,
        df: int = 256,
        n_experts: int = 4,
        topk: int = 1,
        gate_temp: float = 1.2,
        gate_noise: float = 0.0,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        use_gumbel: bool = False,
        use_hard_capacity: bool = True,
        lb_coef: float = 1e-2,
    ):
        super().__init__()
        assert d2d > 0 and d3d > 0
        self.d2d = d2d
        self.d3d = d3d
        self.df = df
        self.K = int(n_experts)  # 每组专家数
        self.topk = max(1, min(int(topk), self.K))
        self.gate_temp = float(gate_temp)
        self.gate_noise = float(gate_noise)
        self.capacity_factor = float(capacity_factor)
        self.use_gumbel = bool(use_gumbel)
        self.use_hard_capacity = bool(use_hard_capacity)
        self.lb_coef = float(lb_coef)

        # 三路输入投影
        self.proj2d = nn.Linear(d2d, df)
        self.proj3d = nn.Linear(d3d, df)
        self.projmix = nn.Linear(d2d + d3d, df)

        # 共享底座（每组各自一套）
        h = max(df // 2, 64)
        self.shared2d = nn.Sequential(nn.LayerNorm(df), nn.GELU(), nn.Linear(df, df))
        self.shared3d = nn.Sequential(nn.LayerNorm(df), nn.GELU(), nn.Linear(df, df))
        self.sharedmix = nn.Sequential(nn.LayerNorm(df), nn.GELU(), nn.Linear(df, df))

        # 组别门控（各自产生一个logit）
        self.group_gate2d = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.group_gate3d = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.group_gatemix = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, 1)
        )

        # 组内专家门控（输出K个expert logits）
        self.expert_gate2d = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, self.K)
        )
        self.expert_gate3d = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, self.K)
        )
        self.expert_gatemix = nn.Sequential(
            nn.LayerNorm(df), nn.Linear(df, h), nn.GELU(), nn.Linear(h, self.K)
        )

        # 三组专家FFN
        self.expert2d = nn.ModuleList([ExpertFFN(df, expansion=4, dropout=dropout) for _ in range(self.K)])
        self.expert3d = nn.ModuleList([ExpertFFN(df, expansion=4, dropout=dropout) for _ in range(self.K)])
        self.expertmix = nn.ModuleList([ExpertFFN(df, expansion=4, dropout=dropout) for _ in range(self.K)])

        # 输出投影
        self.out_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(df, df))

    # --- MoG-inspired utils ---
    def _cv_squared(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        eps = 1e-10
        if x is None:
            return torch.tensor(0.0)
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    # --- helpers (existing code expected below) ---
    def _mask_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        # mask: True for valid positions
        inv = (~mask).unsqueeze(-1)
        logits = logits.masked_fill(inv, float('-inf'))
        return logits

    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_noise <= 0:
            return logits
        noise = torch.randn_like(logits) * self.gate_noise
        return logits + noise

    def _group_weighted_sum(self, xs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], p_group: torch.Tensor) -> torch.Tensor:
        x2d, x3d, xmix = xs
        # p_group: [B, N, 3]
        return (
            x2d * p_group[..., 0].unsqueeze(-1)
            + x3d * p_group[..., 1].unsqueeze(-1)
            + xmix * p_group[..., 2].unsqueeze(-1)
        )

    def _hard_top1_capacity_route(self, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        # Simplified version: dispatch boolean mask per expert, importance and load stats
        B, N, K = logits.shape
        # Add noise and temperature
        logits = self._add_noise(logits) / max(self.gate_temp, 1e-6)
        probs = F.softmax(logits, dim=-1)
        top1 = torch.argmax(probs, dim=-1)  # [B, N]
        dispatch = torch.zeros_like(probs, dtype=torch.bool)
        dispatch.scatter_(-1, top1.unsqueeze(-1), True)
        if mask is not None:
            dispatch = dispatch & mask.unsqueeze(-1)
        # importance: mean prob per expert; load: count of tokens routed
        importance = probs.sum(dim=(0, 1)) / max(B * N, 1)
        load = dispatch.sum(dim=(0, 1)).float() / max(B * N, 1)
        return dispatch, importance, load

    def _topk_soft(self, logits: torch.Tensor):
        # temperature and soft top-k weighting
        logits = self._add_noise(logits) / max(self.gate_temp, 1e-6)
        topk_val, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
        weights = F.softmax(topk_val, dim=-1)
        # scatter back to K
        B, N, K = logits.shape
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, topk_idx, weights)
        return probs, topk_idx

    def forward(self, h2d: torch.Tensor, h3d: torch.Tensor, mask: Optional[torch.Tensor] = None, return_stats: bool = True):
        # 输入投影与共享底座
        base2d = self.shared2d(self.proj2d(h2d))
        base3d = self.shared3d(self.proj3d(h3d))
        basemix = self.sharedmix(self.projmix(torch.cat([h2d, h3d], dim=-1)))

        # 组别门控：三个logit -> softmax
        g2d = self._add_noise(self.group_gate2d(base2d)) / max(self.gate_temp, 1e-6)
        g3d = self._add_noise(self.group_gate3d(base3d)) / max(self.gate_temp, 1e-6)
        gmix = self._add_noise(self.group_gatemix(basemix)) / max(self.gate_temp, 1e-6)
        group_logits = torch.cat([g2d, g3d, gmix], dim=-1)  # [B, N, 3]
        if mask is not None:
            group_logits = self._mask_logits(group_logits, mask)
        p_group = F.softmax(group_logits, dim=-1)
        # z-loss (logit normalization) for stability
        z_loss_group = (group_logits.logsumexp(dim=-1) ** 2).mean()

        if self.use_hard_capacity and self.topk == 1:
            # 2D组
            logits2d = self._mask_logits(self._add_noise(self.expert_gate2d(base2d)), mask)
            dispatch2d, imp2d, load2d = self._hard_top1_capacity_route(logits2d, mask)
            out2d = torch.zeros_like(base2d)
            for k in range(self.K):
                m = dispatch2d[..., k]
                if m.any():
                    y = self.expert2d[k](base2d[m])
                    out2d[m] = y
            out2d = self.out_proj(out2d + base2d)

            # 3D组
            logits3d = self._mask_logits(self._add_noise(self.expert_gate3d(base3d)), mask)
            dispatch3d, imp3d, load3d = self._hard_top1_capacity_route(logits3d, mask)
            out3d = torch.zeros_like(base3d)
            for k in range(self.K):
                m = dispatch3d[..., k]
                if m.any():
                    y = self.expert3d[k](base3d[m])
                    out3d[m] = y
            out3d = self.out_proj(out3d + base3d)

            # 混合组
            logitsmix = self._mask_logits(self._add_noise(self.expert_gatemix(basemix)), mask)
            dispatchmix, impmix, loadmix = self._hard_top1_capacity_route(logitsmix, mask)
            outmix = torch.zeros_like(basemix)
            for k in range(self.K):
                m = dispatchmix[..., k]
                if m.any():
                    y = self.expertmix[k](basemix[m])
                    outmix[m] = y
            outmix = self.out_proj(outmix + basemix)

            out = self._group_weighted_sum([out2d, out3d, outmix], p_group)
            out = out * (mask.unsqueeze(-1).float() if mask is not None else 1.0)

            if return_stats:
                imp_group = p_group.mean(dim=(0, 1))
                lb_loss = self.lb_coef * (
                    self._cv_squared(imp_group)
                    + self._cv_squared(imp2d) + self._cv_squared(load2d)
                    + self._cv_squared(imp3d) + self._cv_squared(load3d)
                    + self._cv_squared(impmix) + self._cv_squared(loadmix)
                )
                stats = {
                    "group_importance": imp_group.detach(),
                    "z_loss_group": z_loss_group.detach(),
                    "expert_importance_2d": imp2d.detach(),
                    "expert_importance_3d": imp3d.detach(),
                    "expert_importance_mix": impmix.detach(),
                    "expert_load_2d": load2d.detach(),
                    "expert_load_3d": load3d.detach(),
                    "expert_load_mix": loadmix.detach(),
                    "lb_loss": lb_loss.detach(),
                }
            return out, stats
        else:
            # 组内 Top-k 软加权
            # 2D组
            logits2d = self._mask_logits(self._add_noise(self.expert_gate2d(base2d)), mask)
            probs2d, _ = self._topk_soft(logits2d)
            y2d = torch.zeros_like(base2d)
            for k in range(self.K):
                yk = self.expert2d[k](base2d)
                p_k = probs2d[..., k].unsqueeze(-1)
                y2d = y2d + p_k * yk
            out2d = self.out_proj(y2d + base2d)

            # 3D组
            logits3d = self._mask_logits(self._add_noise(self.expert_gate3d(base3d)), mask)
            probs3d, _ = self._topk_soft(logits3d)
            y3d = torch.zeros_like(base3d)
            for k in range(self.K):
                yk = self.expert3d[k](base3d)
                p_k = probs3d[..., k].unsqueeze(-1)
                y3d = y3d + p_k * yk
            out3d = self.out_proj(y3d + base3d)

            # 混合组
            logitsmix = self._mask_logits(self._add_noise(self.expert_gatemix(basemix)), mask)
            probsmix, _ = self._topk_soft(logitsmix)
            ymix = torch.zeros_like(basemix)
            for k in range(self.K):
                yk = self.expertmix[k](basemix)
                p_k = probsmix[..., k].unsqueeze(-1)
                ymix = ymix + p_k * yk
            outmix = self.out_proj(ymix + basemix)

            out = self._group_weighted_sum([out2d, out3d, outmix], p_group)
            out = out * (mask.unsqueeze(-1).float() if mask is not None else 1.0)

            if return_stats:
                imp_group = p_group.mean(dim=(0, 1))
                imp2d = probs2d.mean(dim=(0, 1))
                imp3d = probs3d.mean(dim=(0, 1))
                impmix = probsmix.mean(dim=(0, 1))
                entropy_group = (
                    -p_group.clamp_min(1e-8) * p_group.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                lb_loss = self.lb_coef * (
                    self._cv_squared(imp_group)
                    + self._cv_squared(imp2d)
                    + self._cv_squared(imp3d)
                    + self._cv_squared(impmix)
                )
                stats = {
                    "group_importance": imp_group.detach(),
                    "entropy_group": entropy_group.detach(),
                    "z_loss_group": z_loss_group.detach(),
                    "expert_importance_2d": imp2d.detach(),
                    "expert_importance_3d": imp3d.detach(),
                    "expert_importance_mix": impmix.detach(),
                    "lb_loss": lb_loss.detach(),
                }
            return out, stats