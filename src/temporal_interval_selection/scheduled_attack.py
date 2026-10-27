"""Memory-bounded PGD for one locked contiguous edit interval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .temporal import temporal_regularizers


class VideoAdapter(Protocol):
    def get_video_embedding(self, clip: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class PGDConfig:
    """Paper settings for scheduled PGD on one selected interval."""

    device: str = "cuda"
    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 15
    lambda1: float = 0.5
    lambda2: float = 0.0
    final_step_size_ratio: float = 0.1


class AttackSettings(Protocol):
    device: str
    epsilon: float
    alpha: float
    steps: int
    lambda1: float
    lambda2: float
    final_step_size_ratio: float


class RunnerHooks(Protocol):
    def build_sliding_windows(
        self,
        frame_count: int,
        *,
        window_size: int,
        stride: int,
    ) -> Sequence[Sequence[int]]: ...

    def overlapping_window_indices(
        self,
        frame_count: int,
        changed_positions: Sequence[int],
        *,
        window_size: int,
        stride: int,
    ) -> Sequence[int]: ...

    def compute_ssim(self, clean: torch.Tensor, adversarial: torch.Tensor) -> float: ...


def _streaming_max_similarity(
    runtime: RunnerHooks,
    adapter: VideoAdapter,
    query: torch.Tensor,
    windows: Sequence[Sequence[int]],
    video_frames: torch.Tensor,
    changed_offsets: dict[int, int],
    clean_changed: torch.Tensor,
    perturbation: torch.Tensor,
    *,
    gradients: bool,
) -> torch.Tensor:
    best: torch.Tensor | None = None
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context:
        for window in windows:
            frames: list[torch.Tensor] = []
            for position in window:
                offset = changed_offsets.get(position)
                if offset is None:
                    frames.append(video_frames[position].to(query.device))
                else:
                    frames.append(
                        torch.clamp(clean_changed[offset] + perturbation[offset], 0.0, 1.0)
                    )
            clip = torch.stack(frames).unsqueeze(0)
            score = torch.mv(adapter.get_video_embedding(clip), query).squeeze(0)
            if best is None or float(score.detach()) > float(best.detach()):
                best = score
            del clip
    if best is None:
        raise ValueError("no retrieval window overlaps the edited interval")
    return best


def run_scheduled_pgd(
    runtime: RunnerHooks,
    adapter: VideoAdapter,
    video_frames: torch.Tensor,
    query_embedding: torch.Tensor,
    config: AttackSettings,
    *,
    window_size: int,
    stride: int,
    changed_positions: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the fixed-step PGD procedure used for a locked edit interval.

    Only the edited frames are retained as trainable CUDA state. Temporal
    regularizers and reported fidelity metrics are normalized over the full
    input sequence, including zero-perturbation frames outside the interval.
    """
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("scheduled PGD requires CUDA")
    if video_frames.ndim != 4 or not video_frames.is_floating_point():
        raise ValueError("video_frames must be a floating tensor shaped (T, C, H, W)")
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be one-dimensional")
    if window_size < 1 or stride < 1 or stride > window_size:
        raise ValueError("invalid sliding-window configuration")
    if (
        config.steps < 1
        or config.epsilon < 0
        or config.alpha < 0
        or not 0.0 <= config.final_step_size_ratio <= 1.0
    ):
        raise ValueError("invalid PGD configuration")

    frame_count = int(video_frames.shape[0])
    changed = tuple(changed_positions)
    if not changed or any(
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position < frame_count
        for position in changed
    ):
        raise ValueError("changed_positions contains an invalid frame index")
    if changed != tuple(range(changed[0], changed[0] + len(changed))):
        raise ValueError("the editable interval must be contiguous")

    all_windows = runtime.build_sliding_windows(
        frame_count,
        window_size=window_size,
        stride=stride,
    )
    affected_indices = runtime.overlapping_window_indices(
        frame_count,
        changed,
        window_size=window_size,
        stride=stride,
    )
    windows = tuple(all_windows[index] for index in affected_indices)
    if not windows:
        raise ValueError("no retrieval windows overlap the edited interval")

    device_query = query_embedding.detach().to(device)
    changed_offsets = {position: offset for offset, position in enumerate(changed)}
    clean_changed = video_frames[list(changed)].detach().to(device)
    perturbation = torch.zeros_like(clean_changed, requires_grad=True)

    for step in range(config.steps):
        similarity_loss = _streaming_max_similarity(
            runtime,
            adapter,
            device_query,
            windows,
            video_frames,
            changed_offsets,
            clean_changed,
            perturbation,
            gradients=True,
        )
        temporal = temporal_regularizers(
            perturbation,
            total_frames=frame_count,
            changed_positions=changed,
        )
        loss = (
            similarity_loss
            + config.lambda1 * temporal.flicker
            + config.lambda2 * temporal.acceleration
        )
        loss.backward()
        if perturbation.grad is None:
            raise RuntimeError("PGD produced no perturbation gradient")

        decay = 1.0 - config.final_step_size_ratio
        step_size = config.alpha * (1.0 - decay * step / max(config.steps - 1, 1))
        with torch.no_grad():
            perturbation.sub_(step_size * perturbation.grad.sign())
            perturbation.clamp_(-config.epsilon, config.epsilon)
            perturbation.copy_(
                torch.maximum(
                    torch.minimum(perturbation, 1.0 - clean_changed),
                    -clean_changed,
                )
            )
        perturbation.grad = None

    with torch.no_grad():
        adversarial_changed = torch.clamp(clean_changed + perturbation.detach(), 0.0, 1.0)
        similarity_loss = _streaming_max_similarity(
            runtime,
            adapter,
            device_query,
            windows,
            video_frames,
            changed_offsets,
            clean_changed,
            perturbation.detach(),
            gradients=False,
        )
        temporal = temporal_regularizers(
            perturbation.detach(),
            total_frames=frame_count,
            changed_positions=changed,
        )
        squared_error = (adversarial_changed - clean_changed).square().sum().item()
        mse = squared_error / video_frames.numel()
        sequence_psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        mean_changed_ssim = sum(
            runtime.compute_ssim(
                clean_changed[index : index + 1],
                adversarial_changed[index : index + 1],
            )
            for index in range(len(changed))
        ) / len(changed)
        sequence_ssim = (
            frame_count - len(changed) + len(changed) * mean_changed_ssim
        ) / frame_count
        perturbation_linf = perturbation.detach().abs().max().item()
        adversarial_cpu = adversarial_changed.cpu()

    adversarial_video = video_frames.detach().clone()
    adversarial_video[list(changed)] = adversarial_cpu.to(adversarial_video.device)
    metrics: dict[str, Any] = {
        "psnr": sequence_psnr if math.isfinite(sequence_psnr) else None,
        "psnr_reason": None if math.isfinite(sequence_psnr) else "identical tensors",
        "ssim": float(sequence_ssim),
        "perturbation_linf": float(perturbation_linf),
        "flicker": float(temporal.flicker.item()),
        "acceleration": float(temporal.acceleration.item()),
        "loss_sim": float(similarity_loss.item()),
        "iterations": config.steps,
        "changed_frame_count": len(changed),
        "affected_window_count": len(affected_indices),
    }
    return adversarial_video, metrics
