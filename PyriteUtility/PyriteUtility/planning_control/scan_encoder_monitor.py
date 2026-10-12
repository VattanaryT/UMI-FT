"""Inference-time diagnostics for the scan-heatmap CNN branch.

Two independent pieces, both used by ``umift_env_runner.py``:

* :class:`ScanEncoderMonitor` hooks the policy's ``obs_encoder`` (duck-typed:
  ``scan_heatmap_keys`` + ``scan_heatmap_model_map``, see ``TimmObsEncoderWithForce``)
  and, after every policy query, reports statistics that help answer "is the scan
  encoder collapsing?", i.e. producing (nearly) the same feature vector no matter what
  it is shown.
* :func:`render_scan_panel` / :func:`stack_for_display` build the OpenCV popup that puts
  the heatmap (or EE-centered crop) the policy actually saw next to the rgb / depth
  frames.

Nothing here changes policy behaviour: hooks only read tensors.

Reading the numbers (all computed over the last ``window`` policy queries):

``resp_ratio``  (the main collapse indicator)
    L2 spread of the scan feature across the window, divided by how far the feature
    sits from the blank-input feature on average (mean ||f_t - f_blank||). Independent
    of the encoder's constant bias offset. ~1 = the input-driven part of the feature
    changes a lot from query to query; ~0 = the encoder answers (almost) the same thing
    for every input. Below ``resp_ratio_warn`` while the input varies -> ``COLLAPSE?``.
``blank_dist``
    mean-window ||f - f_blank|| / ||f_blank||, where f_blank is the encoder output for
    an all-zero input (what "no scan / flat surface" produces). ~0 means the output is
    just the bias, i.e. the encoder ignores its input -> ``IGNORES-INPUT``.
``rel_spread``
    sqrt(mean per-dim variance) / RMS feature magnitude (informational: it is
    diluted by the constant bias offset, so it reads low even for a working encoder).
``input_spread``
    mean per-pixel std (normalised [0, 1] units) of the heatmap/crop over the window.
    If this is ~0 (blank crop, or stale scan and a barely-moving EE) a constant feature
    is expected and the status is ``input-static`` / ``input-blank``, not a collapse.
``pr``
    participation ratio of the feature covariance (effective number of independent
    directions, <= window - 1). ~1 means the features move along a single direction.
``dead_ch_frac`` / ``active_frac``
    share of conv4 channels that are zero everywhere (dead ReLUs) / of conv4 units that
    are active for the current input.
``cond_scan_energy_ratio``
    (share of the policy's conditioning-vector energy carried by the scan block) /
    (share of its dimensions). ~1 = scan contributes its fair share, << 1 = the other
    modalities dominate the conditioning.
"""

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


class ScanEncoderMonitor:
    def __init__(
        self,
        policy,
        window: int = 200,
        input_window: int = 50,
        min_queries: int = 10,
        resp_ratio_warn: float = 0.1,
        ignore_blank_dist: float = 1e-3,
        rel_spread_warn: float = 0.02,  # fallback when no blank feature is available
        input_spread_min: float = 0.005,
        dead_ch_warn: float = 0.9,
    ):
        self.enabled = False
        self.window = int(window)
        self.min_queries = int(min_queries)
        self.resp_ratio_warn = float(resp_ratio_warn)
        self.ignore_blank_dist = float(ignore_blank_dist)
        self.rel_spread_warn = float(rel_spread_warn)
        self.input_spread_min = float(input_spread_min)
        self.dead_ch_warn = float(dead_ch_warn)

        self._feats: deque = deque(maxlen=self.window)
        self._inputs: deque = deque(maxlen=int(input_window))
        self._cap: Dict[str, torch.Tensor] = {}
        self._handles: List = []
        self._prev_feat: Optional[np.ndarray] = None
        self.n_queries = 0
        self.last_stats: Optional[dict] = None
        self._f_blank: Optional[np.ndarray] = None

        enc = getattr(policy, "obs_encoder", None)
        keys = list(getattr(enc, "scan_heatmap_keys", None) or [])
        if enc is None or len(keys) == 0:
            print("[scan-enc] policy has no scan_heatmap encoder; monitor disabled.")
            return
        self.enc = enc
        self.key = keys[0]
        self.n_keys = len(keys)
        self.scan_model = enc.scan_heatmap_model_map[self.key]
        if len(keys) > 1:
            print(f"[scan-enc] {len(keys)} scan keys; monitoring only '{self.key}'.")

        self._f_blank = self._blank_feature()

        # NOTE: every hook keeps only the *first* call after the obs_encoder started, so a
        # shared scan model used for several keys still reports the first (monitored) key.
        def _clear(_m, _args):
            self._cap.clear()

        self._handles.append(enc.register_forward_pre_hook(_clear))
        self._handles.append(
            self.scan_model.register_forward_pre_hook(
                lambda m, args: self._cap.setdefault("input", args[0].detach())
            )
        )
        self._handles.append(
            self.scan_model.backbone.register_forward_hook(
                lambda m, args, out: self._cap.setdefault("conv", out.detach())
            )
        )
        self._handles.append(
            self.scan_model.register_forward_hook(
                lambda m, args, out: self._cap.setdefault("feat", out.detach())
            )
        )
        self._handles.append(
            enc.register_forward_hook(
                lambda m, args, out: self._cap.__setitem__("cond", out.detach())
            )
        )
        self.enabled = True
        print(
            f"[scan-enc] monitoring '{self.key}' "
            f"(pooling={getattr(self.scan_model, 'pooling', '?')}, "
            f"feature_dim={getattr(self.scan_model, 'feature_dim', '?')})"
        )

    # ------------------------------------------------------------------ helpers
    def _blank_feature(self) -> Optional[np.ndarray]:
        """Encoder output for an all-zero input (the 'no data / flat surface' feature)."""
        try:
            shape = tuple(self.enc.key_shape_map[self.key])  # (C, H, W)
            device = next(self.scan_model.parameters()).device
            with torch.no_grad():
                x = torch.zeros((1, *shape), device=device, dtype=torch.float32)
                return self.scan_model(x)[0].float().cpu().numpy()
        except Exception as exc:  # diagnostics must never break inference
            print(f"[scan-enc] could not compute blank feature: {exc}")
            return None

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.enabled = False

    def reset(self) -> None:
        """Forget history (call at the start of an episode)."""
        self._feats.clear()
        self._inputs.clear()
        self._cap.clear()
        self._prev_feat = None
        self.n_queries = 0
        self.last_stats = None

    # ------------------------------------------------------------------ main API
    def update(self) -> Optional[dict]:
        """Consume the tensors captured during the last ``policy.predict_action`` call."""
        if not self.enabled:
            return None
        cap = self._cap
        if "input" not in cap or "feat" not in cap:
            return None

        x = cap["input"][-1].float().cpu().numpy()  # newest frame (C, H, W)
        feat_all = cap["feat"].float().cpu().numpy()  # (B*T, D)
        f = feat_all[-1]
        self.n_queries += 1

        x0 = x[0]
        self._inputs.append(x0.astype(np.float32, copy=True))
        self._feats.append(f.astype(np.float32, copy=True))

        stats: dict = {
            "n": self.n_queries,
            "in_nonzero_frac": float((x0 > 0).mean()),
            "in_mean": float(x0.mean()),
            "feat_norm": float(np.linalg.norm(f)),
        }

        if self._prev_feat is not None:
            denom = np.linalg.norm(f) * np.linalg.norm(self._prev_feat) + 1e-12
            stats["cos_prev"] = float(np.dot(f, self._prev_feat) / denom)
            # cos_prev prints as 1.000 for tiny changes; this keeps the resolution.
            stats["d_prev"] = float(
                np.linalg.norm(f - self._prev_feat) / (np.linalg.norm(self._prev_feat) + 1e-12)
            )
        else:
            stats["cos_prev"] = float("nan")
            stats["d_prev"] = float("nan")
        self._prev_feat = f.copy()

        if self._f_blank is not None:
            stats["blank_dist"] = float(
                np.linalg.norm(f - self._f_blank)
                / (np.linalg.norm(self._f_blank) + 1e-8)
            )
        else:
            stats["blank_dist"] = float("nan")

        # spread over recent queries
        F = np.stack(self._feats)  # (n, D)
        if len(F) >= 2:
            spread = float(np.sqrt(F.var(axis=0).mean()))
            scale = float(np.sqrt((F**2).mean()))
            stats["rel_spread"] = spread / (scale + 1e-8)
        else:
            stats["rel_spread"] = float("nan")
        if self._f_blank is not None and len(F) >= 2:
            R = float(np.linalg.norm(F - self._f_blank, axis=1).mean())
            spread_l2 = float(np.sqrt(F.var(axis=0).sum()))
            stats["blank_dist_win"] = R / (float(np.linalg.norm(self._f_blank)) + 1e-8)
            stats["resp_ratio"] = spread_l2 / (R + 1e-8)
        else:
            stats["blank_dist_win"] = float("nan")
            stats["resp_ratio"] = float("nan")
        if len(F) >= 8:
            s = np.linalg.svd(F - F.mean(axis=0), compute_uv=False)
            lam = s**2
            stats["pr"] = float(lam.sum() ** 2 / (np.sum(lam**2) + 1e-12))
        else:
            stats["pr"] = float("nan")
        if len(self._inputs) >= 2:
            stats["input_spread"] = float(np.stack(self._inputs).std(axis=0).mean())
        else:
            stats["input_spread"] = float("nan")

        # conv4 activations: dead channels / active units
        if "conv" in cap:
            act = cap["conv"][-1].float().cpu().numpy()  # (C, h, w)
            on = act > 0
            stats["active_frac"] = float(on.mean())
            stats["dead_ch_frac"] = float((~on.reshape(on.shape[0], -1).any(axis=1)).mean())
        else:
            stats["active_frac"] = float("nan")
            stats["dead_ch_frac"] = float("nan")

        # how much of the conditioning vector is the scan block?
        if "cond" in cap and cap["cond"].dim() == 2:
            cond = cap["cond"].float().cpu().numpy()
            B, D = cond.shape
            per_key = (feat_all.shape[0] // max(B, 1)) * feat_all.shape[1]
            block = min(self.n_keys * per_key, D)
            c = cond[0]
            blk = c[D - block : D - block + min(per_key, block)]
            share = float((blk**2).sum() / ((c**2).sum() + 1e-12))
            frac = float(len(blk) / D)
            stats["cond_scan_share"] = share
            stats["cond_scan_energy_ratio"] = share / (frac + 1e-12)
        else:
            stats["cond_scan_share"] = float("nan")
            stats["cond_scan_energy_ratio"] = float("nan")

        stats["status"] = self._status(stats)
        stats["feature"] = f
        self.last_stats = stats
        return stats

    def _status(self, s: dict) -> str:
        if s["n"] < self.min_queries:
            return f"warming-up ({s['n']}/{self.min_queries})"
        if s["in_nonzero_frac"] == 0.0 and not (s["input_spread"] > 0):
            return "input-blank"
        if s["input_spread"] < self.input_spread_min:
            return "input-static"
        if s["dead_ch_frac"] >= self.dead_ch_warn:
            return "DEAD-RELU?"
        if np.isfinite(s["resp_ratio"]):
            if s["blank_dist_win"] < self.ignore_blank_dist:
                return "IGNORES-INPUT"
            if s["resp_ratio"] < self.resp_ratio_warn:
                return "COLLAPSE?"
        elif s["rel_spread"] < self.rel_spread_warn:  # no blank reference available
            return "COLLAPSE?"
        return "ok"

    # ------------------------------------------------------------------ reporting
    @staticmethod
    def _f(v, fmt="{:.3f}") -> str:
        return "n/a" if v is None or not np.isfinite(v) else fmt.format(v)

    def format_line(self, s: dict) -> str:
        f = self._f
        return (
            f"[scan-enc] #{s['n']} {s['status']} | "
            f"resp_ratio={f(s['resp_ratio'])} in_spread={f(s['input_spread'], '{:.4f}')} "
            f"blank_dist={f(s['blank_dist'])} rel_spread={f(s['rel_spread'])} "
            f"cos_prev={f(s['cos_prev'])} d_prev={f(s['d_prev'], '{:.1e}')} "
            f"pr={f(s['pr'], '{:.1f}')} dead_ch={f(s['dead_ch_frac'], '{:.2f}')} "
            f"cond_energy_x={f(s['cond_scan_energy_ratio'], '{:.2f}')} "
            f"in_nonzero={f(s['in_nonzero_frac'], '{:.2f}')}"
        )

    def overlay_lines(self, s: dict) -> List[str]:
        f = self._f
        return [
            f"enc: {s['status']}",
            f"resp_ratio {f(s['resp_ratio'])}  blank {f(s['blank_dist'])}",
            f"in_spread {f(s['input_spread'], '{:.4f}')}  nz {f(s['in_nonzero_frac'], '{:.2f}')}",
            f"d_prev {f(s['d_prev'], '{:.1e}')}  pr {f(s['pr'], '{:.1f}')}",
        ]

    def summary(self) -> str:
        if self.last_stats is None:
            return "[scan-enc] no queries recorded."
        return "[scan-enc] episode summary: " + self.format_line(self.last_stats)[len("[scan-enc] ") :]

    @staticmethod
    def log_scalars(s: dict) -> Dict[str, Optional[float]]:
        """Numeric subset of ``update()``'s output, suitable for (JSON) zarr attrs.

        Non-finite values (e.g. cos_prev on the first query) become None.
        """
        return {
            k: (float(v) if np.isfinite(v) else None)
            for k, v in s.items()
            if k not in ("status", "feature") and np.isscalar(v)
        }


# ---------------------------------------------------------------------- display
def _to_bgr_u8(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im)
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[..., 0]
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(im)


def stack_for_display(images: List[np.ndarray], height: Optional[int] = None) -> np.ndarray:
    """hstack frames of mixed channels / heights into one BGR uint8 image."""
    imgs = [_to_bgr_u8(im) for im in images]
    h = int(height) if height is not None else imgs[0].shape[0]
    out = []
    for im in imgs:
        if im.shape[0] != h:
            w = max(int(round(im.shape[1] * h / im.shape[0])), 1)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(im)
    return np.hstack(out)


def render_scan_panel(
    scan_obs: np.ndarray,
    height: int,
    crop: bool = False,
    text_lines: Optional[List[str]] = None,
) -> np.ndarray:
    """Colour-mapped view of the heatmap exactly as the policy receives it.

    scan_obs: (T, C, H, W) or (C, H, W), normalised to [0, 1] (0 = flat / no data,
    1 = clip value). Channel 0 is drawn with JET (blue = 0 .. red = clip). When ``crop``
    is True a cross marks the centre of the patch, which is the tool position.
    """
    arr = np.asarray(scan_obs)
    if arr.ndim == 4:
        arr = arr[-1]
    ch0 = np.clip(arr[0], 0.0, 1.0)
    img = cv2.applyColorMap((ch0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h, w = img.shape[:2]
    scale = height / float(h)
    img = cv2.resize(
        img, (max(int(round(w * scale)), 1), int(height)), interpolation=cv2.INTER_NEAREST
    )
    H, W = img.shape[:2]
    if crop:
        cx, cy, r = W // 2, H // 2, max(H // 24, 4)
        cv2.line(img, (cx - r, cy), (cx + r, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy - r), (cx, cy + r), (255, 255, 255), 1)
    font_scale = max(H / 900.0, 0.4)
    y = int(24 * font_scale / 0.5)
    for line in ["scan crop" if crop else "scan heatmap"] + list(text_lines or []):
        for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(
                img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick,
                cv2.LINE_AA,
            )
        y += int(26 * font_scale / 0.5)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (255, 255, 255), 1)
    return img
