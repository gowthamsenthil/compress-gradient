"""GPUWorker abstraction: an MNIST MLP replica + a network profile."""
from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .compression import ErrorFeedback, topk_compress, topk_compress_per_layer
from .cost_model import NetworkProfile


class MLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def flatten_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().view(-1) for p in model.parameters()])


def unflatten_into_grads(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_layer_offsets(model: nn.Module) -> list[tuple[int, int]]:
    """Group parameters by their owning module (one "layer" per sub-module with
    trainable params). Returns list of (start, end) indices into the flat
    gradient vector, in the same order as `model.parameters()`.

    For the MLP this yields 3 layers (fc1, fc2, fc3), each covering its
    weight + bias slice.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for module in model.modules():
        # Only leaf modules with their own parameters.
        own_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if not own_params:
            continue
        start = cursor
        for p in own_params:
            cursor += p.numel()
        offsets.append((start, cursor))
    return offsets


def layer_bits(model: nn.Module, bits_per_param: int = 32) -> list[int]:
    """Total bits per layer, in the same order as get_layer_offsets."""
    return [(e - s) * bits_per_param for s, e in get_layer_offsets(model)]


class GPUWorker:
    """Simulated worker: model replica + data shard + network profile."""

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        loader: DataLoader,
        profile: NetworkProfile,
        device: torch.device | str = "cpu",
        use_error_feedback: bool = True,
    ):
        self.id = worker_id
        self.model = model.to(device)
        self.loader = loader
        self.profile = profile
        self.device = device
        self._iter: Optional[Iterator] = None
        self.ef = ErrorFeedback(num_params(self.model), device=device) if use_error_feedback else None
        # Precompute per-layer offsets so per-layer compression / alpha
        # calibration is cheap.
        self.layer_offsets = get_layer_offsets(self.model)

    def _next_batch(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)

    def compute_gradient(self) -> torch.Tensor:
        """Forward + backward on the next mini-batch; returns flat grad."""
        self.model.train()
        x, y = self._next_batch()
        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return flatten_grads(self.model)

    def compress(self, g: torch.Tensor, ratio) -> tuple[torch.Tensor, float]:
        """Compress gradient and report the squared error that actually leaked.

        `ratio` can be a scalar (one ratio for the whole gradient) or a 1-D
        iterable of per-layer ratios (length == len(self.layer_offsets)).

        With error feedback, the leaked error is ||(g + residual_prev) - sent||^2,
        which is in general smaller than ||g - TopK(g)||^2 because EF folds in
        accumulated residuals. We measure it at the call site so logging is
        EF-correct.
        """
        per_layer = hasattr(ratio, "__len__")

        if self.ef is not None:
            corrected = g + self.ef.residual
            if per_layer:
                sent = topk_compress_per_layer(corrected, ratio, self.layer_offsets)
            else:
                sent = topk_compress(corrected, float(ratio))
            new_residual = corrected - sent
            self.ef.residual = new_residual
            leaked_sq = float(new_residual.pow(2).sum().item())
            return sent, leaked_sq

        if per_layer:
            sent = topk_compress_per_layer(g, ratio, self.layer_offsets)
        else:
            sent = topk_compress(g, float(ratio))
        leaked_sq = float((g - sent).pow(2).sum().item())
        return sent, leaked_sq

    def compute_layer_alphas(self, g: Optional[torch.Tensor] = None) -> "list[float]":
        """Per-layer ||g_l||^2. If `g` is None, computes a fresh gradient."""
        if g is None:
            g = self.compute_gradient()
        return [float(g[s:e].pow(2).sum().item()) for s, e in self.layer_offsets]

    def compute_per_layer_leaked_sq(self, residual_or_error: torch.Tensor) -> "list[float]":
        """Split a squared-error/residual vector per layer."""
        return [float(residual_or_error[s:e].pow(2).sum().item())
                for s, e in self.layer_offsets]

    def apply_gradient(self, flat_grad: torch.Tensor, lr: float) -> None:
        with torch.no_grad():
            offset = 0
            for p in self.model.parameters():
                n = p.numel()
                p.data.add_(flat_grad[offset : offset + n].view_as(p), alpha=-lr)
                offset += n

    def sync_weights_from(self, source_state_dict) -> None:
        self.model.load_state_dict(source_state_dict)
