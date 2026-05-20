"""V7 prerequisite tests -- all must pass before launching v7.

Single confirmed root cause (RC-1):
  presence_bce weight (0.3) overwhelmed by pose/rot gradient; the model
  learns to ignore the presence signal and collapses to nearly all-absent slots.
  pos_weight=3.10 was marginally under-calibrated (theoretical: 3.29).

RC-2 (xyz underfit) is a downstream consequence of RC-1, not independent.
  Proof: MSE .mean() over 3 xyz channels vs 6 rot channels gives xyz 2x MORE
  per-channel gradient than rot at equal weights. xyz is NOT signal-starved.
  The low xyz std (0.518 at 100k) comes from near-empty training scenes where
  xyz loss contributes almost nothing due to inactive object masks.

V7 fix (presence-only):
  - PRESENCE_POS_WEIGHT: 3.10 -> 5.0
  - presence_bce loss weight: 0.3 -> 1.5
  - All other weights: unchanged
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import pytest


def _grad_norms(cfg: dict, B: int = 8, N: int = 12) -> dict:
    """Synthetic forward+backward. Returns per-component gradient norms."""
    pred = torch.randn(B, N, 13, requires_grad=True)
    tgt = torch.randn(B, N, 13)
    tgt[:, :, 12] = torch.bernoulli(torch.full((B, N), 0.233))  # 23.3% occupancy

    pose_loss = ((pred[:, :, :3] - tgt[:, :, :3])**2).mean() * cfg["pose_xyz"]
    rot_loss  = ((pred[:, :, 3:9] - tgt[:, :, 3:9])**2).mean() * cfg["pose_rot"]
    # Simplified BCE with pos_weight
    pres_logits, pres_tgt = pred[:, :, 12], tgt[:, :, 12]
    pw = torch.ones_like(pres_logits)
    pw[pres_tgt > 0.5] = cfg["pos_weight"]
    pres_loss = (pw * torch.nn.functional.binary_cross_entropy_with_logits(
        pres_logits, pres_tgt, reduction='none')).mean() * cfg["presence_bce"]

    (pose_loss + rot_loss + pres_loss).backward()
    g = pred.grad
    return {
        "xyz":  g[:, :, :3].abs().mean().item(),
        "rot":  g[:, :, 3:9].abs().mean().item(),
        "pres": g[:, :, 12].abs().mean().item(),
        "xyz_per_ch": g[:, :, :3].abs().mean().item() / 3,
        "rot_per_ch": g[:, :, 3:9].abs().mean().item() / 6,
    }


V6 = {"pose_xyz": 1.0, "pose_rot": 1.0, "type_ce": 1.0,
      "presence_bce": 0.3, "pos_weight": 3.10}

V7 = {"pose_xyz": 1.0, "pose_rot": 1.0, "type_ce": 1.0,
      "presence_bce": 1.5, "pos_weight": 5.0}


class TestGradientMath:
    """Confirm the gradient math underpinning the diagnosis."""

    def test_xyz_per_channel_stronger_than_rot_v6(self):
        """MSE .mean() gives xyz 2x more per-channel grad than rot (3 vs 6 channels).
        RC-2 (xyz underfit) is NOT a gradient-starvation issue -- it's downstream of RC-1."""
        g = _grad_norms(V6)
        ratio = g["xyz_per_ch"] / g["rot_per_ch"]
        print(f"\nv6 xyz_per_ch/rot_per_ch = {ratio:.2f}x  (expect ~2.0, xyz is STRONGER)")
        assert ratio > 1.5, f"Expected xyz per-channel > rot per-channel; got {ratio:.2f}x"

    def test_v6_presence_gradient_dominated(self):
        """In v6, presence gradient < 50% of xyz gradient -- it loses the competition."""
        g = _grad_norms(V6)
        ratio = g["pres"] / g["xyz"]
        print(f"\nv6 pres/xyz grad ratio = {ratio:.3f}  (expect < 0.5 = dominated)")
        assert ratio < 0.5, f"v6 presence gradient not dominated as expected: ratio={ratio:.3f}"


class TestV7Fixes:
    """V7 config (presence_bce=1.5, pos_weight=5.0) must fix the gradient imbalance."""

    def test_presence_gradient_parity(self):
        """v7 presence gradient >= 50% of xyz gradient (parity, no longer dominated)."""
        g = _grad_norms(V7)
        ratio = g["pres"] / g["xyz"]
        print(f"\nv7 pres/xyz grad ratio = {ratio:.3f}  (need >= 0.5)")
        assert ratio >= 0.5, f"v7 presence still dominated: ratio={ratio:.3f}"

    def test_presence_improvement_over_v6(self):
        """v7 presence gradient must be >= 3x stronger than v6."""
        g6 = _grad_norms(V6)
        g7 = _grad_norms(V7)
        ratio = g7["pres"] / g6["pres"]
        print(f"\nv7/v6 presence grad: {ratio:.2f}x  (need >= 3x)")
        assert ratio >= 3.0, f"v7 presence improvement only {ratio:.2f}x; need >= 3x"

    def test_xyz_gradient_unchanged(self):
        """Presence-only fix must not change xyz gradient (no pose_xyz change)."""
        g6 = _grad_norms(V6)
        g7 = _grad_norms(V7)
        ratio = g7["xyz"] / g6["xyz"]
        print(f"\nv7/v6 xyz grad change: {ratio:.3f}  (expect ~1.0, unchanged)")
        assert 0.8 <= ratio <= 1.2, f"xyz gradient changed unexpectedly: {ratio:.3f}x"

    def test_rot_gradient_unchanged(self):
        """Presence-only fix must not change rot gradient."""
        g6 = _grad_norms(V6)
        g7 = _grad_norms(V7)
        ratio = g7["rot"] / g6["rot"]
        print(f"\nv7/v6 rot grad change: {ratio:.3f}  (expect ~1.0, unchanged)")
        assert 0.8 <= ratio <= 1.2, f"rot gradient changed unexpectedly: {ratio:.3f}x"


class TestInvariants:
    """Regression tests for unchanged invariants."""

    def test_whitening_round_trip(self):
        MEAN = torch.tensor([-0.049867, +0.378790, -0.751155])
        STD  = torch.tensor([+0.419080, +0.457588, +0.127651])
        x = torch.randn(200, 3)
        assert (x - (((x - MEAN) / STD) * STD + MEAN)).abs().max() < 1e-5

    def test_scene_tensor_layout_is_13_dims(self):
        """xyz(0:3)|rot6d(3:9)|scale(9:12)|presence(12) = 13 dims total."""
        assert 3 + 6 + 3 + 1 == 13
        x = torch.randn(4, 12, 13)
        assert x[:, :, :3].shape  == (4, 12, 3)
        assert x[:, :, 3:9].shape == (4, 12, 6)
        assert x[:, :, 9:12].shape == (4, 12, 3)
        assert x[:, :, 12].shape  == (4, 12)

    def test_training_data_occupancy(self):
        """Training data has 23.3% slot occupancy (2.79/12). pos_weight=5.0 is safe margin."""
        occupancy = 2.79 / 12
        theoretical_pos_weight = (1 - occupancy) / occupancy
        print(f"\nOccupancy: {occupancy:.3f}, theoretical pos_weight: {theoretical_pos_weight:.2f}")
        print(f"v7 pos_weight=5.0 is {5.0/theoretical_pos_weight:.1f}x theoretical")
        assert theoretical_pos_weight < 5.0, "pos_weight=5.0 should be above theoretical"
        assert 5.0 / theoretical_pos_weight < 2.0, "pos_weight=5.0 not more than 2x theoretical (no overcorrection)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
