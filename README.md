# Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval

Reference implementation for the paper **“Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval”**, accepted as a full paper at SOICT 2026.

## Overview

This repository implements the interval-selection methods evaluated in the paper and the scheduled PGD attack described in the paper:
- target-centered;
- highest-score;
- minimum-unaffected-floor; and
- Gradient-LSE.

The default setup uses 16 keyframes, stride 1, an `ℓ∞` budget of 8/255, and 15 PGD updates. Frozen settings are in `configs/paper.json`.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install .
```

## Usage

```python
from temporal_interval_selection import analyze_intervals, run_scheduled_pgd, PGDConfig

selection = analyze_intervals(
    clean_window_scores,
    windows,
    frame_count=len(video_frames),
    target_positions=target_positions,
    window_size=16,
)

adversarial_video, metrics = run_scheduled_pgd(
    runtime, adapter, video_frames, query_embedding, PGDConfig(),
    window_size=16, stride=1,
    changed_positions=selection.minimum_unaffected_floor.positions,
)
```

Inputs are `(T, C, H, W)` RGB tensors in `[0, 1]`. The caller provides the model adapter, video data, retrieval windows, and corpus-level evaluation hooks.

## Repository structure

```text
├── configs/
│   └── paper.json                    Frozen window, budget, PGD, and utility settings
├── docs/
│   ├── data-format.md                JSONL authority formats and metric assumptions
│   ├── paper-alignment-audit.md      Paper-to-code alignment review
│   └── reproduction.md               Cohorts, required assets, and run order
├── scripts/
│   └── rebuild_fidelity_report.py    CLI wrapper for per-query PSNR/SSIM recovery
├── src/
│   └── temporal_interval_selection/
│       ├── selectors.py              Three deterministic interval selectors
│       ├── scheduled_attack.py       Memory-bounded PGD and run metrics
│       ├── temporal.py               Full-sequence first/second-order penalties
│       ├── fidelity.py               Sequence-to-edited-frame metric conversion
│       ├── results.py                Authority validation and aggregation
│       └── gradient_lse.py           Gradient-LSE selector
└── tests/
    └── test_cpu_smoke.py             CPU semantic smoke tests
```

## Reproduction

The package contains the algorithmic core and fidelity reconstruction. Reproducing the
paper tables also requires the locked queries and targets, 873-video corpus and cache,
exact SigLIP runtime/checkpoint, full-corpus reranking, and per-query authorities.

Those artifacts are external to this repository. See [docs/reproduction.md](docs/reproduction.md)
for the expected run order and input formats.
