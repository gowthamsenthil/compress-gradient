"""Top-K sparsification with optional error feedback."""
from __future__ import annotations

import torch


def topk_compress(g: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep the `ratio` fraction of coordinates with largest magnitude; zero rest.

    Args:
        g: 1-D gradient tensor.
        ratio: in (0, 1]. ratio=1.0 means no compression.
    Returns:
        Sparse-style tensor with same shape as g (zeros in dropped positions).
    """
    if ratio >= 1.0:
        return g.clone()
    k = max(1, int(round(ratio * g.numel())))
    out = torch.zeros_like(g)
    _, idx = torch.topk(g.abs(), k, sorted=False)
    out[idx] = g[idx]
    return out


class ErrorFeedback:
    """Maintains residual error per worker. Enables convergence with biased
    compressors (Top-K) per Karimireddy et al. 2019."""

    def __init__(self, num_params: int, device: torch.device | str = "cpu"):
        self.residual = torch.zeros(num_params, device=device)

    def compress(self, g: torch.Tensor, ratio: float) -> torch.Tensor:
        corrected = g + self.residual
        sent = topk_compress(corrected, ratio)
        self.residual = corrected - sent
        return sent

    def reset(self):
        self.residual.zero_()


def squared_error(g: torch.Tensor, ratio: float) -> float:
    """Empirical ||g - TopK(g)||^2 for a given ratio."""
    return float((g - topk_compress(g, ratio)).pow(2).sum().item())


def topk_compress_per_layer(
    g: torch.Tensor,
    ratios: "list | tuple",
    offsets: "list[tuple[int, int]]",
) -> torch.Tensor:
    """Apply Top-K independently to each contiguous layer slice of a flat grad.

    Args:
        g: 1-D flat gradient.
        ratios: iterable of per-layer ratios, length == len(offsets).
        offsets: list of (start, end) indices, one per layer; contiguous and
            covering [0, g.numel()).
    Returns:
        Flat tensor with per-layer Top-K applied; same shape as g.
    """
    if len(ratios) != len(offsets):
        raise ValueError(f"len(ratios)={len(ratios)} != len(offsets)={len(offsets)}")
    out = torch.zeros_like(g)
    for (s, e), r in zip(offsets, ratios):
        out[s:e] = topk_compress(g[s:e], float(r))
    return out
