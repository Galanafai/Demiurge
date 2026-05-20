"""Oracle tests that must all pass before training launches.

Maps directly to past failure modes:
  test_normalize_denormalize_identity  -> would catch double-denorm in probe
  test_whitening_round_trip            -> would catch inv-whitening sign flip
  test_zero_terminal_snr               -> confirms v5 schedule patch applied
  test_v_prediction_round_trip         -> confirms v-pred math is correct
  test_whitening_constants_consistent  -> train.py and probe must agree
  test_gradscaler_api                  -> catches deprecated cuda.amp.GradScaler
  test_pytorch_cuda_works              -> basic sanity
  test_invariants_module               -> TrainingInvariants logic works
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, pytest


def test_pytorch_cuda_works():
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(64, 12, 13, device="cuda")
    y = (x @ x.transpose(-1, -2))
    assert not torch.isnan(y).any()
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU: {torch.cuda.get_device_name(0)}, {mem_gb:.1f} GB")


def test_normalize_denormalize_identity():
    """Round-trip must be identity within float32 precision."""
    from scene.schema import SceneTensor, WorkspaceBounds
    bounds = WorkspaceBounds.default()
    # Build a random but valid SceneTensor
    poses = torch.zeros(12, 7); poses[:, 3] = 1.0  # valid quaternion w=1
    poses[:5, 0] = torch.tensor([0.3, -0.2, 0.4, 0.1, -0.3])   # x in workspace
    poses[:5, 1] = torch.tensor([0.3,  0.4, 0.2, 0.5,  0.1])   # y
    poses[:5, 2] = torch.tensor([0.05, 0.08, 0.06, 0.07, 0.05]) # z (table height)
    scales = torch.ones(12, 3) * 0.05
    presence = torch.tensor([True]*5 + [False]*7)
    otypes = torch.zeros(12, dtype=torch.long)
    scene = SceneTensor(object_types=otypes, poses=poses, scales=scales, presence=presence)
    
    norm = scene.normalize(bounds)
    recovered = norm.denormalize(bounds)
    
    diff = (scene.poses[:5] - recovered.poses[:5]).abs().max().item()
    print(f"\nnormalize-denormalize round-trip error: {diff:.2e}")
    assert diff < 1e-5, f"Round-trip error {diff} exceeds tolerance"


def test_whitening_round_trip():
    """whiten then unwhiten must recover original normalized values."""
    from scene.typed_scene import NormalizedScene, WhitenedScene
    from scene.schema import SceneTensor
    
    poses = torch.randn(12, 7) * 0.3
    poses[:, 3:] = 0.0; poses[:, 3] = 1.0  # unit quaternion
    poses[:, :3] = poses[:, :3].clamp(-1, 1)
    scene_norm = NormalizedScene(SceneTensor(
        object_types=torch.zeros(12, dtype=torch.long),
        poses=poses, scales=torch.randn(12, 3)*0.2,
        presence=torch.ones(12, dtype=torch.bool),
    ))
    mean_xyz   = torch.tensor([-0.049867, +0.378790, -0.751155])
    std_xyz    = torch.tensor([+0.419080, +0.457588, +0.127651])
    mean_scale = torch.tensor([-0.038706, -0.038706, -0.038706])
    std_scale  = torch.tensor([+0.221619, +0.221619, +0.221619])
    
    whitened  = scene_norm.whiten(mean_xyz, std_xyz, mean_scale, std_scale)
    recovered = whitened.unwhiten(mean_xyz, std_xyz, mean_scale, std_scale)
    
    # After clamping in unwhiten, values within [-1,1] must match
    orig_xyz = scene_norm.inner.poses[:, :3].clamp(-1, 1)
    recv_xyz = recovered.inner.poses[:, :3]
    diff = (orig_xyz - recv_xyz).abs().max().item()
    print(f"\nwhitening round-trip xyz error: {diff:.2e}")
    assert diff < 1e-5, f"Whitening round-trip failed: {diff}"


def test_zero_terminal_snr():
    """alpha_bar[T] must be exactly 0 when zero_terminal_snr=True."""
    from model.schedule import CosineSchedule
    s = CosineSchedule(T=1000, zero_terminal_snr=True)
    ab_T = float(s._alpha_bar[-1])
    print(f"\nalpha_bar[-1] = {ab_T:.4e}")
    assert ab_T < 1e-10, f"zero_terminal_snr failed: alpha_bar[-1]={ab_T}"
    
    # Standard schedule should NOT be zero
    s2 = CosineSchedule(T=1000, zero_terminal_snr=False)
    # Standard cosine naturally gives ~2.4e-9 (very small but non-zero float)
    # The key distinction: zero_terminal_snr=True gives EXACTLY 0.0 (integer zero)
    assert float(s2._alpha_bar[-1]) != 0.0, "Standard schedule alpha_bar[-1] should be non-zero float"


def test_v_prediction_round_trip():
    """compute_v_target -> predict_x0_from_v must recover x0."""
    from model.schedule import CosineSchedule
    s = CosineSchedule(T=1000, zero_terminal_snr=True)
    
    x0  = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    t   = torch.randint(0, 1000, (4,))
    
    v      = s.compute_v_target(x0, eps, t)
    x_t, _ = s.add_noise(x0, t, eps)
    x0_hat = s.predict_x0_from_v(x_t, t, v)
    
    err = (x0_hat - x0).abs().max().item()
    print(f"\nv-prediction round-trip error: {err:.2e}")
    assert err < 1e-4, f"v-prediction round-trip failed: {err}"


def test_whitening_constants_consistent():
    """train.py and probe_drake.py must use identical whitening constants."""
    import re
    train_src = open("scripts/train.py").read()
    probe_src  = open("scripts/probe_drake.py").read()
    
    assert "_DATA_MEAN_XYZ" in train_src, "train.py missing _DATA_MEAN_XYZ"
    assert "_DATA_STD_XYZ"  in train_src, "train.py missing _DATA_STD_XYZ"
    assert "_DATA_MEAN_XYZ" in probe_src, "probe_drake.py missing _DATA_MEAN_XYZ"
    assert "_DATA_STD_XYZ"  in probe_src, "probe_drake.py missing _DATA_STD_XYZ"
    
    # Extract values and compare
    def extract_tensor(src, name):
        m = re.search(rf'{name}\s*=\s*(?:torch|_torch)\.tensor\(\[([^\]]+)\]', src)
        return [float(x.strip()) for x in m.group(1).split(",")] if m else None
    
    t_mean = extract_tensor(train_src, "_DATA_MEAN_XYZ")
    p_mean = extract_tensor(probe_src,  "_DATA_MEAN_XYZ")
    print(f"\ntrain mean: {t_mean}")
    print(f"probe mean: {p_mean}")
    assert t_mean == p_mean, f"Mismatch: train={t_mean} probe={p_mean}"
    
    t_std = extract_tensor(train_src, "_DATA_STD_XYZ")
    p_std = extract_tensor(probe_src,  "_DATA_STD_XYZ")
    assert t_std == p_std, f"Std mismatch: train={t_std} probe={p_std}"


def test_gradscaler_api():
    """GradScaler must use new torch.amp API, not deprecated torch.cuda.amp."""
    src = open("scripts/train.py").read()
    assert "torch.cuda.amp.GradScaler" not in src, \
        "train.py uses deprecated torch.cuda.amp.GradScaler -- must use torch.amp.GradScaler('cuda')"
    assert "torch.amp.GradScaler" in src, \
        "train.py missing torch.amp.GradScaler"


def test_v5_config_values():
    """v5 config must have prediction_type=v and zero_terminal_snr=true."""
    import yaml
    cfg = yaml.safe_load(open("configs/train/v9_uncond_v5.yaml"))
    assert cfg["diffusion"]["prediction_type"] == "v", "prediction_type must be 'v'"
    assert cfg["diffusion"]["zero_terminal_snr"] == True, "zero_terminal_snr must be true"
    print(f"\nConfig: {cfg['diffusion']}")


def test_invariants_module():
    """TrainingInvariants catches all bug classes correctly."""
    from training.invariants import TrainingInvariants
    
    inv = TrainingInvariants()
    
    # NaN loss -> halt
    assert not inv.check_loss({"total": float("nan")}, step=0)
    assert inv.should_halt()
    
    # Reset and test type_ce == 0 at step 1000
    inv2 = TrainingInvariants()
    assert inv2.check_loss({"total": 1.5, "type_ce": 0.0}, step=1000) == False
    
    # Output collapse at step 10001
    inv3 = TrainingInvariants()
    collapsed = torch.zeros(64, 12, 13)  # all zeros = std=0 -> collapse
    assert inv3.check_output(collapsed, step=10001) == False
    assert "collapsed" in inv3.last_violation()
    
    # Clean case
    inv4 = TrainingInvariants()
    good_out = torch.randn(64, 12, 13)
    assert inv4.check_loss({"total": 1.5, "type_ce": 0.3}, step=1000) == True
    assert inv4.check_output(good_out, step=10001) == True
    assert not inv4.should_halt()
    print("\nAll invariant checks pass")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
