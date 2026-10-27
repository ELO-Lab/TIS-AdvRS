"""Temporal regularizers for a contiguous, partially editable video interval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class TemporalLoss:
    """First- and second-order temporal penalties."""

    flicker: torch.Tensor
    acceleration: torch.Tensor


def temporal_regularizers(
    perturbation: torch.Tensor,
    *,
    total_frames: int,
    changed_positions: Sequence[int],
) -> TemporalLoss:
    """Compute full-sequence temporal losses from a compact edited-frame tensor.

    ``perturbation`` has shape ``(K, C, H, W)`` and stores only the edited
    interval. Frames outside that interval are treated as zero perturbations.
    The denominators therefore use the full sequence length, matching the
    manuscript's normalization by ``(T - 1) C H W`` and ``(T - 2) C H W``.
    """
    if perturbation.ndim != 4:
        raise ValueError("perturbation must have shape (K, C, H, W)")
    if min(perturbation.shape) < 1:
        raise ValueError("perturbation dimensions must be non-zero")
    if isinstance(total_frames, bool) or not isinstance(total_frames, int):
        raise TypeError("total_frames must be an integer")
    if total_frames < 1:
        raise ValueError("total_frames must be positive")

    changed = tuple(changed_positions)
    if len(changed) != perturbation.shape[0]:
        raise ValueError("changed_positions length must equal perturbation frame count")
    if not changed:
        raise ValueError("at least one frame must be editable")
    if any(isinstance(position, bool) or not isinstance(position, int) for position in changed):
        raise TypeError("changed_positions must contain integers")
    if changed != tuple(range(changed[0], changed[0] + len(changed))):
        raise ValueError("changed_positions must be sorted and contiguous")
    if changed[0] < 0 or changed[-1] >= total_frames:
        raise ValueError("changed_positions fall outside the full sequence")

    pixels_per_frame = prod(perturbation.shape[1:])
    zero = perturbation.new_zeros(())

    first_order_sum = zero
    if len(changed) > 1:
        first_order_sum = (perturbation[1:] - perturbation[:-1]).square().sum()
    if changed[0] > 0:
        first_order_sum = first_order_sum + perturbation[0].square().sum()
    if changed[-1] < total_frames - 1:
        first_order_sum = first_order_sum + perturbation[-1].square().sum()

    flicker = first_order_sum / max((total_frames - 1) * pixels_per_frame, 1)

    acceleration_sum = zero
    if total_frames > 2:
        local_offset = {position: offset for offset, position in enumerate(changed)}
        first_center = max(0, changed[0] - 2)
        last_center = min(total_frames - 3, changed[-1])
        for center in range(first_center, last_center + 1):
            curvature: torch.Tensor | None = None
            for position, coefficient in (
                (center, 1.0),
                (center + 1, -2.0),
                (center + 2, 1.0),
            ):
                offset = local_offset.get(position)
                if offset is None:
                    continue
                term = coefficient * perturbation[offset]
                curvature = term if curvature is None else curvature + term
            if curvature is not None:
                acceleration_sum = acceleration_sum + curvature.square().sum()

    acceleration = acceleration_sum / max(
        (total_frames - 2) * pixels_per_frame,
        1,
    )
    return TemporalLoss(flicker=flicker, acceleration=acceleration)
