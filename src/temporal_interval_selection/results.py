"""Per-query recovery of edited-frame fidelity from saved run authorities."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .fidelity import recover_edited_fidelity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"authority file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _index_by_query(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"missing query_id in {source}")
        if query_id in indexed:
            raise ValueError(f"duplicate query_id {query_id!r} in {source}")
        indexed[query_id] = row
    return indexed


def _frame_counts(interval_authority: Path, *, edited_frames: int) -> dict[str, int]:
    rows = _index_by_query(_read_jsonl(interval_authority), source=interval_authority)
    counts: dict[str, int] = {}
    for query_id, row in rows.items():
        analysis = row.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError(f"missing interval analysis for {query_id}")
        intervals = analysis.get("intervals_evaluated")
        if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 1:
            raise ValueError(f"invalid intervals_evaluated for {query_id}")
        # A length-K interval has T-K+1 valid start positions.
        counts[query_id] = intervals + edited_frames - 1
    return counts


def _sequence_metrics(
    row: Mapping[str, Any], *, query_id: str, method: str, edited_frames: int
) -> tuple[float | None, float]:
    if row.get("status") != "success":
        raise ValueError(f"{method}/{query_id} is not a successful authority row")
    measurements = row.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError(f"{method}/{query_id} has no measurements object")
    if measurements.get("changed_frame_count") != edited_frames:
        raise ValueError(f"{method}/{query_id} did not edit exactly {edited_frames} frames")
    if measurements.get("uses_mock") is True:
        raise ValueError(f"{method}/{query_id} is marked as a mock run")

    psnr = measurements.get("psnr")
    ssim = measurements.get("ssim")
    if psnr is not None and (
        isinstance(psnr, bool)
        or not isinstance(psnr, (int, float))
        or not math.isfinite(psnr)
    ):
        raise ValueError(f"invalid sequence PSNR for {method}/{query_id}")
    if (
        isinstance(ssim, bool)
        or not isinstance(ssim, (int, float))
        or not math.isfinite(ssim)
    ):
        raise ValueError(f"invalid sequence SSIM for {method}/{query_id}")
    return psnr, float(ssim)


def rebuild_fidelity_report(
    interval_authority: Path,
    method_authorities: Mapping[str, Path],
    *,
    edited_frames: int = 16,
) -> dict[str, Any]:
    """Recover per-query edited-frame fidelity, then average across queries.

    Inputs must contain pooled full-sequence MSE-derived PSNR and uniformly
    frame-averaged SSIM. Unedited frames are assumed to be unchanged. Sequence
    lengths are recovered from the count of valid contiguous intervals.
    """
    if isinstance(edited_frames, bool) or not isinstance(edited_frames, int) or edited_frames < 1:
        raise ValueError("edited_frames must be a positive integer")
    if not method_authorities:
        raise ValueError("at least one method authority is required")
    if any(not name.strip() for name in method_authorities):
        raise ValueError("method names cannot be empty")

    total_frames = _frame_counts(Path(interval_authority), edited_frames=edited_frames)
    if not total_frames:
        raise ValueError("interval authority contains no queries")

    report: dict[str, Any] = {
        "query_count": len(total_frames),
        "edited_frame_count": edited_frames,
        "aggregation": "recover per query, then take the arithmetic mean",
        "methods": {},
    }
    for method, raw_path in method_authorities.items():
        path = Path(raw_path)
        rows = _index_by_query(_read_jsonl(path), source=path)
        if set(rows) != set(total_frames):
            raise ValueError(f"query IDs in {method} authority do not match interval authority")

        per_query: list[dict[str, Any]] = []
        for query_id in sorted(rows):
            sequence_psnr, sequence_ssim = _sequence_metrics(
                rows[query_id], query_id=query_id, method=method, edited_frames=edited_frames
            )
            length = total_frames[query_id]
            edited = recover_edited_fidelity(
                sequence_psnr,
                sequence_ssim,
                total_frames=length,
                edited_frames=edited_frames,
            )
            per_query.append(
                {
                    "query_id": query_id,
                    "total_frames": length,
                    "sequence_psnr_db": sequence_psnr,
                    "sequence_ssim": sequence_ssim,
                    "edited_psnr_db": edited.psnr_db,
                    "edited_ssim": edited.ssim,
                }
            )

        finite_psnr = [
            row["edited_psnr_db"]
            for row in per_query
            if row["edited_psnr_db"] is not None
        ]
        report["methods"][method] = {
            "mean_edited_psnr_db_finite_only": (
                statistics.mean(finite_psnr) if finite_psnr else None
            ),
            "infinite_psnr_count": len(per_query) - len(finite_psnr),
            "mean_edited_ssim": statistics.mean(row["edited_ssim"] for row in per_query),
            "per_query": per_query,
        }
    return report
