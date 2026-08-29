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

    Architecture: a few stride-2 Conv2d+ReLU layers followed by global average
    pooling and a linear projection to a fixed-size feature vector. Global average
    pooling (rather than a flatten + fixed-size Linear, as used in the
    `multimodal_representation` DepthEncoder/ImageEncoder this mirrors) makes the
    module agnostic to the exact input H/W, which is convenient since scan_heatmap
    resolution is a task-config knob.

    Input:  (N, in_channels, H, W) float tensor (any H, W; downsampled 16x total
            by four stride-2 convs, then globally pooled).
    Output: (N, feature_dim) float tensor.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"ScanHeatmapEncoder expects (N,C,H,W), got {tuple(x.shape)}"
        assert x.shape[1] == self.in_channels, (
            f"ScanHeatmapEncoder configured for {self.in_channels} input channels, "
            f"got {x.shape[1]}"
        )
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)
