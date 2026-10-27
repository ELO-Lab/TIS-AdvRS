"""Command-line interface for per-query fidelity recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .results import rebuild_fidelity_report


def _method_authorities(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise argparse.ArgumentTypeError("authorities must use METHOD=FILE format")
        if name in result:
            raise argparse.ArgumentTypeError(f"duplicate method authority: {name}")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover edited-frame PSNR and SSIM for each query, then summarize."
    )
    parser.add_argument("--interval-authority", type=Path, required=True)
    parser.add_argument(
        "--authority",
        action="append",
        required=True,
        metavar="METHOD=FILE",
        help="Per-query full-sequence authority; repeat once for each method.",
    )
    parser.add_argument("--edited-frames", type=int, default=16)
    parser.add_argument("--output", type=Path, help="Write JSON here; defaults to stdout.")
    args = parser.parse_args()

    report = rebuild_fidelity_report(
        args.interval_authority,
        _method_authorities(args.authority),
        edited_frames=args.edited_frames,
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Wrote fidelity report to {args.output}")
