"""Conversions between whole-sequence and edited-keyframe fidelity metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EditedFidelity:
    """Fidelity scores for the edited frames of one query."""

    psnr_db: float | None
    ssim: float


def recover_edited_fidelity(
    sequence_psnr_db: float | None,
    sequence_ssim: float,
    *,
    total_frames: int,
    edited_frames: int,
) -> EditedFidelity:
    """Recover edited-frame metrics from one query's sequence-level metrics.

    The PSNR conversion assumes sequence MSE is pooled over every pixel in the
    video. The SSIM conversion assumes frame-level SSIM is averaged uniformly
    and unchanged frames have SSIM equal to one. Call this once per query,
    before averaging across a cohort.
    """
    for value, name in ((total_frames, "total_frames"), (edited_frames, "edited_frames")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if edited_frames > total_frames:
        raise ValueError("edited_frames cannot exceed total_frames")
    if not math.isfinite(sequence_ssim) or not -1.000001 <= sequence_ssim <= 1.000001:
        raise ValueError("sequence_ssim must be finite and within [-1, 1]")
    if sequence_psnr_db is not None and not math.isfinite(sequence_psnr_db):
        raise ValueError("non-finite sequence PSNR must be represented as None")

    edited_psnr_db = (
        None
        if sequence_psnr_db is None
        else sequence_psnr_db - 10.0 * math.log10(total_frames / edited_frames)
    )
    edited_ssim = 1.0 + (sequence_ssim - 1.0) * total_frames / edited_frames
    if not -1.000001 <= edited_ssim <= 1.000001:
        raise ValueError("recovered edited-frame SSIM falls outside [-1, 1]")

    return EditedFidelity(psnr_db=edited_psnr_db, ssim=edited_ssim)
