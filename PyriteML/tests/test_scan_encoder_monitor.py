"""Tests for the inference-time scan-encoder monitor and popup helpers."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.vision.scan_heatmap_encoder import ScanHeatmapEncoder
from PyriteUtility.planning_control.scan_encoder_monitor import (
    ScanEncoderMonitor,
    render_scan_panel,
    stack_for_display,
)


class FakeObsEncoder(nn.Module):
    """Mimics TimmObsEncoderWithForce's scan branch: other features + scan feature last."""

    def __init__(self, hw=(64, 64), pooling="flatten", other_dim=40):
        super().__init__()
        self.scan_heatmap_keys = ["scan_heatmap_0"]
        self.key_shape_map = {"scan_heatmap_0": (3, *hw)}
        self.scan_heatmap_model_map = nn.ModuleDict(
            {"scan_heatmap_0": ScanHeatmapEncoder(3, 128, pooling=pooling, input_hw=hw)}
        )
        self.other_dim = other_dim

    def forward(self, obs):
        x = obs["scan_heatmap_0"]  # (B, T, C, H, W)
        B, T = x.shape[:2]
        f = self.scan_heatmap_model_map["scan_heatmap_0"](x.reshape(B * T, *x.shape[2:]))
        other = torch.ones(B, self.other_dim)
        return torch.cat([other, f.reshape(B, -1)], dim=-1)


class FakePolicy(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.obs_encoder = FakeObsEncoder(**kw)

    def predict(self, x):
        with torch.no_grad():
            return self.obs_encoder({"scan_heatmap_0": x})


def _blob(rng, hw=(64, 64)):
    img = np.zeros((1, 1, 3, *hw), np.float32)  # (B, T, C, H, W)
    cy, cx = rng.integers(8, hw[0] - 8), rng.integers(8, hw[1] - 8)
    img[..., cy - 4 : cy + 4, cx - 4 : cx + 4] = rng.uniform(0.3, 1.0)
    return torch.from_numpy(img)


def test_monitor_reports_healthy_encoder():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pol = FakePolicy()
    mon = ScanEncoderMonitor(pol, min_queries=5)
    assert mon.enabled
    stats = None
    for _ in range(12):
        pol.predict(_blob(rng))
        stats = mon.update()
    assert stats["n"] == 12
    assert stats["status"] == "ok", stats
    assert stats["resp_ratio"] > 0.1
    assert stats["blank_dist_win"] > 0.01
    assert stats["pr"] > 1.5
    assert stats["feature"].shape == (128,)
    # scan block is the last 128 of 168 conditioning dims
    assert 0.0 < stats["cond_scan_share"] < 1.0
    assert stats["cond_scan_energy_ratio"] > 0.0
    assert "resp_ratio" in mon.format_line(stats) and "d_prev" in mon.format_line(stats)
    assert stats["d_prev"] > 0


def test_monitor_flags_collapsed_encoder():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    pol = FakePolicy()
    # collapse: zero all conv/proj weights -> output is just the bias, whatever the input
    for p in pol.obs_encoder.scan_heatmap_model_map.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        pol.obs_encoder.scan_heatmap_model_map["scan_heatmap_0"].proj.bias.copy_(
            torch.randn(128)
        )
    mon = ScanEncoderMonitor(pol, min_queries=5)
    for _ in range(12):
        pol.predict(_blob(rng))
        stats = mon.update()
    assert stats["rel_spread"] < 1e-6
    assert stats["blank_dist"] < 1e-6
    assert stats["input_spread"] > 0.005  # input does vary...
    # ...but the feature never moves (all conv channels are dead / output is the bias)
    assert stats["status"] in ("IGNORES-INPUT", "DEAD-RELU?")


def test_monitor_blank_input_is_not_called_a_collapse():
    torch.manual_seed(0)
    pol = FakePolicy()
    mon = ScanEncoderMonitor(pol, min_queries=3)
    for _ in range(6):
        pol.predict(torch.zeros(1, 1, 3, 64, 64))
        stats = mon.update()
    assert stats["status"] == "input-blank"
    assert stats["blank_dist"] == pytest.approx(0.0, abs=1e-6)


def test_monitor_disabled_without_scan_branch_and_reset():
    class NoScan(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder = nn.Identity()

    assert not ScanEncoderMonitor(NoScan()).enabled

    pol = FakePolicy(pooling="avg")
    mon = ScanEncoderMonitor(pol)
    pol.predict(_blob(np.random.default_rng(2)))
    assert mon.update()["n"] == 1
    mon.reset()
    assert mon.last_stats is None and mon.n_queries == 0
    assert mon.update() is None  # nothing captured since reset
    mon.detach()
    assert not mon.enabled


def test_log_scalars_json_safe():
    pol = FakePolicy()
    mon = ScanEncoderMonitor(pol)
    pol.predict(_blob(np.random.default_rng(3)))
    s = ScanEncoderMonitor.log_scalars(mon.update())
    assert s["cos_prev"] is None  # first query: nan -> None
    assert all(v is None or isinstance(v, float) for v in s.values())


def test_stack_for_display_mixed_channels_and_heights():
    rgb = np.zeros((48, 64, 3), np.uint8)
    depth = np.zeros((48, 64), np.uint8)
    other = np.zeros((24, 32, 3), np.uint8)
    out = stack_for_display([rgb, depth, other])
    assert out.shape == (48, 64 * 2 + 64, 3) and out.dtype == np.uint8


def test_render_scan_panel_shapes():
    crop = np.random.default_rng(0).uniform(0, 1, (1, 3, 64, 64)).astype(np.float32)
    p = render_scan_panel(crop, height=240, crop=True, text_lines=["a", "b"])
    assert p.shape == (240, 240, 3) and p.dtype == np.uint8
    full = np.random.default_rng(0).uniform(0, 1, (1, 3, 80, 160)).astype(np.float32)
    p = render_scan_panel(full, height=240)
    assert p.shape == (240, 480, 3)
    # blue (0) vs red (1) mapping
    z = render_scan_panel(np.zeros((3, 8, 8), np.float32), height=64)
    o = render_scan_panel(np.ones((3, 8, 8), np.float32), height=64)
    assert z[32, 48, 0] > z[32, 48, 2]  # BGR: blue dominates at 0
    assert o[32, 48, 2] > o[32, 48, 0]  # red dominates at 1
