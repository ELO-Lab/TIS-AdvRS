"""Deterministic selectors for a fixed-length contiguous edit interval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectedInterval:
    """One contiguous edit interval and its clean-score diagnostic."""

    start: int
    positions: tuple[int, ...]
    unaffected_score_floor: float | None
    unaffected_count: int


@dataclass(frozen=True)
class IntervalAnalysis:
    """Outputs of the three frozen primary selectors for one query."""

    target_centered: SelectedInterval
    highest_score_window: SelectedInterval
    minimum_unaffected_floor: SelectedInterval
    intervals_evaluated: int


def target_centered_positions(
    frame_count: int,
    target_position: int,
    *,
    window_size: int = 16,
) -> tuple[int, ...]:
    """Return the clamped interval centered on a zero-based target frame.

    For an even interval, the target is the last frame in the left half. A
    16-frame interval therefore places seven frames before and eight frames
    after the target when the video boundary permits it.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if frame_count < window_size or window_size < 1:
        raise ValueError("frame_count must be at least the positive window_size")
    if (
        isinstance(target_position, bool)
        or not isinstance(target_position, int)
        or not 0 <= target_position < frame_count
    ):
        raise ValueError("target_position must be a valid zero-based frame index")

    final_start = frame_count - window_size
    start = min(max(target_position - (window_size - 1) // 2, 0), final_start)
    return tuple(range(start, start + window_size))


def evaluate_interval(
    scores: Sequence[float],
    windows: Sequence[Sequence[int]],
    *,
    start: int,
    frame_count: int,
    window_size: int,
) -> SelectedInterval:
    """Measure the highest clean score in windows unaffected by an interval."""
    if window_size < 1 or frame_count < window_size:
        raise ValueError("invalid frame_count or window_size")
    if not 0 <= start <= frame_count - window_size:
        raise ValueError("interval start falls outside the valid video range")
    if len(scores) != len(windows):
        raise ValueError("scores and windows must have the same length")

    end = start + window_size
    unaffected_scores = [
        float(score)
        for score, window in zip(scores, windows, strict=True)
        if window[-1] < start or window[0] >= end
    ]
    return SelectedInterval(
        start=start,
        positions=tuple(range(start, end)),
        unaffected_score_floor=max(unaffected_scores, default=None),
        unaffected_count=len(unaffected_scores),
    )


def analyze_intervals(
    scores: Sequence[float],
    windows: Sequence[Sequence[int]],
    *,
    frame_count: int,
    target_positions: Sequence[int],
    window_size: int = 16,
) -> IntervalAnalysis:
    """Select target-centered, highest-score, and minimum-floor intervals.

    All selectors share one fixed contiguous interval budget. Ties resolve to
    the earliest interval or window, matching the locked experiment protocol.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if frame_count < window_size or window_size < 1:
        raise ValueError("frame_count must be at least the positive window_size")
    if len(scores) == 0 or len(windows) == 0:
        raise ValueError("at least one scored window is required")
    if len(scores) != len(windows):
        raise ValueError("scores and windows must have the same length")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("window scores must be finite")
    normalized_windows = tuple(tuple(window) for window in windows)
    for window in normalized_windows:
        if len(window) != window_size:
            raise ValueError("every scored window must match window_size")
        if any(isinstance(position, bool) or not isinstance(position, int) for position in window):
            raise TypeError("window positions must be integers")
        if window != tuple(range(window[0], window[0] + window_size)):
            raise ValueError("scored windows must contain contiguous frame positions")
        if window[0] < 0 or window[-1] >= frame_count:
            raise ValueError("a scored window falls outside the video")

    valid_starts = range(frame_count - window_size + 1)
    candidates = [
        evaluate_interval(
            scores,
            normalized_windows,
            start=start,
            frame_count=frame_count,
            window_size=window_size,
        )
        for start in valid_starts
    ]
    minimum_floor = min(
        candidates,
        key=lambda item: (
            -math.inf if item.unaffected_score_floor is None else item.unaffected_score_floor,
            item.start,
        ),
    )

    top_window_index = max(range(len(scores)), key=lambda index: float(scores[index]))
    highest_start = min(normalized_windows[top_window_index][0], frame_count - window_size)

    target = tuple(target_positions)
    if len(target) != window_size:
        raise ValueError("target_positions must contain exactly window_size frames")
    if any(isinstance(position, bool) or not isinstance(position, int) for position in target):
        raise TypeError("target positions must be integers")
    target_start = min(target)
    if target != tuple(range(target_start, target_start + window_size)):
        raise ValueError("target-centered positions must be sorted and contiguous")

    return IntervalAnalysis(
        target_centered=evaluate_interval(
            scores,
            normalized_windows,
            start=target_start,
            frame_count=frame_count,
            window_size=window_size,
        ),
        highest_score_window=evaluate_interval(
            scores,
            normalized_windows,
            start=highest_start,
            frame_count=frame_count,
            window_size=window_size,
        ),
        minimum_unaffected_floor=minimum_floor,
        intervals_evaluated=len(candidates),
    )
