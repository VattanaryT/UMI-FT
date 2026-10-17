import torch
import torch.nn as nn


class ScanHeatmapEncoder(nn.Module):
    """Small, dedicated, from-scratch CNN encoder for the scan_heatmap_0 modality.

    scan_heatmap_0 is a laser-scan-derived surface-deviation heatmap: low spatial
    resolution (e.g. 80x160), low information density relative to full RGB, and
    updated only periodically (~every 30s), with horizon=1 (single most-recent
    frame, no temporal history). It does NOT share the shape/horizon assumptions
    that `TimmObsEncoderWithForce`'s shared pretrained-backbone rgb path hardwires
    (a single common (H, W) across all rgb-typed keys, and a single shared horizon
    used to size the modality-attention feature count). Rather than bending those
    shared assumptions (used by every task in this codebase) or paying for a full
    pretrained ViT/ResNet backbone on this small single-channel-derived signal,
    scan_heatmap_0 gets this tiny, purpose-built CNN branch instead, kept entirely
    separate from the shared vision encoder.

    Architecture: a few stride-2 Conv2d+ReLU layers followed by a pooling step and a
    linear projection to a fixed-size feature vector.

    pooling:
        "avg":     global average pooling (default, original behaviour). Agnostic to
                   the input H/W, but it throws away *where* in the image a feature
                   is, so it can only describe "what is somewhere in the image".
        "flatten": flatten the final (64, H/16, W/16) feature map into the linear
                   layer. Keeps spatial layout, which is required for the
                   EE-centered crop (the policy has to know whether a ridge is
                   ahead of / behind / beside the tool). Needs ``input_hw``.
        "spatial_softmax": learned keypoints (Levine et al. 2016). A 1x1 conv maps the
                   feature map to ``num_keypoints`` channels; each channel gets a
                   softmax over (H', W') (temperature is learnable) and its expected
                   (x, y) in [-1, 1] is the output, i.e. 2*num_keypoints coordinates.
                   The global-average-pooled features are concatenated so the
                   magnitude information (which softmax normalizes away) is kept.
                   Much smaller than flatten and sub-cell precise. Options:
                   ``num_keypoints`` (32), ``include_avg`` (True) and
                   ``init_temperature`` (0.1). Uses 3 stride-2
                   convs + 1 stride-1 conv by default (feature map = input/8, same
                   receptive field as the 4-stride backbone).

    Input:  (N, in_channels, H, W) float tensor (downsampled by 2**num_downsample;
            num_downsample defaults to 4, or 3 for "spatial_softmax").
    Output: (N, feature_dim) float tensor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128,
        pooling: str = "avg",
        input_hw=None,
        num_downsample=None,
        num_keypoints=None,
        include_avg: bool = True,
        init_temperature: float = 0.1,
    ):
        super().__init__()
        assert pooling in (
            "avg",
            "flatten",
            "spatial_softmax",
        ), f"unknown pooling {pooling}"
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.pooling = pooling
        if num_downsample is None:
            num_downsample = 3 if pooling == "spatial_softmax" else 4
        num_downsample = int(num_downsample)
        assert 1 <= num_downsample <= 4, "num_downsample must be in [1, 4]"
        self.num_downsample = num_downsample

        # (out_channels, kernel_size); every one of these is stride 2. The first
        # four entries reproduce the original backbone (state-dict compatible).
        specs = [(16, 5), (32, 3), (64, 3), (64, 3)][:num_downsample]
        layers = []
        c_in = in_channels
        for c_out, k in specs:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=2, padding=k // 2),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        if pooling == "spatial_softmax":
            # stride-1 conv restores receptive field lost by dropping one stride
            layers += [
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)
        self.out_channels = c_in

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
            proj_in = c_in
        elif pooling == "flatten":
            assert (
                input_hw is not None
            ), "ScanHeatmapEncoder(pooling='flatten') needs input_hw=(H, W)"
            self.pool = nn.Flatten(1)
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, int(input_hw[0]), int(input_hw[1]))
                proj_in = int(self.backbone(dummy).numel())
        else:
            self.num_keypoints = int(num_keypoints) if num_keypoints else 32
            self.include_avg = bool(include_avg)
            self.keypoint_conv = nn.Conv2d(c_in, self.num_keypoints, kernel_size=1)
            # softmax(logits / temperature); learnable. Freshly initialised conv logits
            # are tiny, so at temperature 1 the softmax is ~uniform and every keypoint
            # sits near (0, 0); a lower start (0.1) makes the coordinates informative
            # from the first step while still being soft enough to train.
            self.temperature = nn.Parameter(torch.full((1,), float(init_temperature)))
            self.pool = nn.AdaptiveAvgPool2d(1)
            proj_in = 2 * self.num_keypoints + (c_in if self.include_avg else 0)
        self.proj = nn.Linear(proj_in, feature_dim)

    def _spatial_softmax(self, feat: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, 2*K) expected keypoint coordinates in [-1, 1]
        (x = column direction first, then y = row direction)."""
        n, _, h, w = feat.shape
        logits = self.keypoint_conv(feat).float().flatten(2)  # (N, K, H*W), fp32
        attn = torch.softmax(logits / self.temperature.float(), dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        px = (attn * grid_x.reshape(1, 1, -1)).sum(-1)
        py = (attn * grid_y.reshape(1, 1, -1)).sum(-1)
        return torch.cat([px, py], dim=-1).to(feat.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"ScanHeatmapEncoder expects (N,C,H,W), got {tuple(x.shape)}"
        assert x.shape[1] == self.in_channels, (
            f"ScanHeatmapEncoder configured for {self.in_channels} input channels, "
            f"got {x.shape[1]}"
        )
        feat = self.backbone(x)
        if self.pooling == "spatial_softmax":
            out = self._spatial_softmax(feat)
            if self.include_avg:
                out = torch.cat([out, self.pool(feat).flatten(1)], dim=-1)
            return self.proj(out)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)
