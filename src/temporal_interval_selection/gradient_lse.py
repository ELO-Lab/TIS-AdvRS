"""Gradient-LSE interval selection over cached keyframe embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


WindowScorer = Callable[[torch.Tensor, Sequence[int]], torch.Tensor]


@dataclass(frozen=True)
class GradientLSESelection:
    """Selected interval and the per-keyframe gradient saliency."""

    start: int
    positions: tuple[int, ...]
    interval_score: float
    frame_saliency: tuple[float, ...]
    intervals_evaluated: int


def select_gradient_lse_interval(
    frame_embeddings: torch.Tensor,
    windows: Sequence[Sequence[int]],
    score_window: WindowScorer,
    *,
    interval_length: int = 16,
    temperature: float = 0.01,
) -> GradientLSESelection:
    """Select the interval with the largest summed LSE-gradient saliency.

    ``score_window`` returns the clean query-video similarity for one window
    as a differentiable scalar function of ``frame_embeddings``. The
    temperature-scaled log-sum-exp is differentiated with respect to the
    normalized keyframe embeddings. Absolute gradient values are summed over
    feature dimensions, then over each candidate interval. Ties choose the
    earliest interval.
    """
    if frame_embeddings.ndim != 2 or not frame_embeddings.is_floating_point():
        raise ValueError("frame_embeddings must be a floating tensor shaped (T, D)")
    if frame_embeddings.shape[0] < 1 or frame_embeddings.shape[1] < 1:
        raise ValueError("frame_embeddings must have non-zero dimensions")
    if isinstance(interval_length, bool) or not isinstance(interval_length, int):
        raise TypeError("interval_length must be an integer")
    if interval_length < 1 or interval_length > frame_embeddings.shape[0]:
        raise ValueError("interval_length must be between 1 and the frame count")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not windows:
        raise ValueError("at least one retrieval window is required")

    frame_count = int(frame_embeddings.shape[0])
    normalized_windows = tuple(tuple(window) for window in windows)
    for window in normalized_windows:
        if not window:
            raise ValueError("retrieval windows cannot be empty")
        if any(isinstance(position, bool) or not isinstance(position, int) for position in window):
            raise TypeError("window positions must be integers")
        if window != tuple(range(window[0], window[0] + len(window))):
            raise ValueError("retrieval windows must be contiguous")
        if window[0] < 0 or window[-1] >= frame_count:
            raise ValueError("a retrieval window falls outside the video")

    embeddings = frame_embeddings.detach().clone().requires_grad_(True)
    scores = tuple(score_window(embeddings, window) for window in normalized_windows)
    if any(not isinstance(score, torch.Tensor) or score.ndim != 0 for score in scores):
        raise ValueError("score_window must return a scalar tensor")
    score_tensor = torch.stack(scores)
    if not bool(torch.isfinite(score_tensor).all()):
        raise ValueError("window scores must be finite")

    smooth_score = float(temperature) * torch.logsumexp(
        score_tensor / float(temperature), dim=0
    )
    gradient = torch.autograd.grad(smooth_score, embeddings)[0]
    saliency = gradient.abs().sum(dim=-1)
    candidates = torch.stack(
        [
            saliency[start : start + interval_length].sum()
            for start in range(frame_count - interval_length + 1)
        ]
    )
    start = int(candidates.argmax().item())
    return GradientLSESelection(
        start=start,
        positions=tuple(range(start, start + interval_length)),
        interval_score=float(candidates[start].detach().item()),
        frame_saliency=tuple(float(value) for value in saliency.detach().cpu().tolist()),
        intervals_evaluated=len(candidates),
    )
