"""Runtime invariants checked every N steps during training.

Each check corresponds to a bug class observed in previous runs:
  - NaN/Inf loss: gradient explosion, bad LR
  - type_ce == 0: class weight bug, CE not backpropagating (v1 failure)
  - presence logit unbounded: BCE without proper initialization (v2 failure)
  - per-dim std collapse: z-head collapse seen in v4 at 30k steps
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class TrainingInvariants:
    violations: list = field(default_factory=list)

    def check_loss(self, loss_dict: dict, step: int) -> bool:
        total = loss_dict.get("total", 0.0)
        if isinstance(total, torch.Tensor): total = total.item()
        if total != total:   # NaN
            self.violations.append((step, f"loss/total is NaN")); return False
        if total == float('inf'):
            self.violations.append((step, f"loss/total is Inf")); return False
        if step >= 1000:
            type_ce = loss_dict.get("type_ce", 0.0)
            if isinstance(type_ce, torch.Tensor): type_ce = type_ce.item()
            if type_ce < 0.01:
                self.violations.append((step, f"type_ce={type_ce:.5f} -- not backpropagating")); return False
        return True

    def check_output(self, pred_cont: torch.Tensor, step: int) -> bool:
        """pred_cont: (B, N_MAX, 13) model output in whitened space."""
        if torch.isnan(pred_cont).any():
            self.violations.append((step, "model output contains NaN")); return False
        if step >= 5000:
            pres = pred_cont[:, :, 12]
            if pres.abs().max().item() > 15.0:
                self.violations.append((step, f"presence logit |max|={pres.abs().max():.1f} -- unbounded")); return False
        if step >= 10000:
            per_dim_std = pred_cont.std(dim=[0, 1])
            min_std, worst = per_dim_std.min().item(), per_dim_std.argmin().item()
            if min_std < 0.02:
                names = ["x","y","z","r1","r2","r3","r4","r5","r6","sx","sy","sz","pres"]
                self.violations.append((step, f"dim '{names[worst]}' collapsed: std={min_std:.4f}")); return False
        return True

    def should_halt(self) -> bool: return bool(self.violations)
    def last_violation(self) -> str:
        return f"step {self.violations[-1][0]}: {self.violations[-1][1]}" if self.violations else ""


def log_output_stats(pred_cont: torch.Tensor, step: int, run=None) -> dict:
    """Log per-dim mean/std/min/max to W&B. Catches collapse early."""
    names = ["x","y","z","r1","r2","r3","r4","r5","r6","sx","sy","sz","pres"]
    stats = {}
    for i, name in enumerate(names):
        d = pred_cont[:, :, i]
        stats[f"dim_stats/{name}_mean"] = d.mean().item()
        stats[f"dim_stats/{name}_std"]  = d.std().item()
        stats[f"dim_stats/{name}_min"]  = d.min().item()
        stats[f"dim_stats/{name}_max"]  = d.max().item()
    if run is not None:
        run.log(stats, step=step)
    return stats
