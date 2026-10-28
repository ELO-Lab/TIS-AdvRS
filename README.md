# Temporal Interval Selection for Adversarial Rank Suppression

Reference implementation for the SOICT 2026 paper on adversarial rank suppression in
windowed text-to-video retrieval.

## Overview

This repository implements four interval selectors and the scheduled PGD attack used by
the paper:

- target-centered;
- highest-score;
- minimum-unaffected-floor; and
- Gradient-LSE.

The default setup uses 16 keyframes, stride 1, an `ℓ∞` budget of 8/255, and 15 PGD
updates. Frozen settings are in `configs/paper.json`.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install .
python -m unittest discover -s tests -v
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

Inputs are `(T, C, H, W)` RGB tensors in `[0, 1]`. The caller provides the model
adapter, video data, retrieval windows, and corpus-level evaluation hooks.

## Repository structure

```text
configs/
  paper.json
docs/
  data-format.md
  paper-alignment-audit.md
  reproduction.md
scripts/
  rebuild_fidelity_report.py
src/temporal_interval_selection/
  selectors.py
  scheduled_attack.py
  temporal.py
  fidelity.py
  results.py
  gradient_lse.py
tests/
  test_cpu_smoke.py
```

## Reproduction

The package contains the algorithmic core and fidelity reconstruction. Reproducing the
paper tables also requires the locked queries and targets, 873-video corpus and cache,
exact SigLIP runtime/checkpoint, full-corpus reranking, and per-query authorities.

Those artifacts are external to this repository. See [docs/reproduction.md](docs/reproduction.md)
for the expected run order and input formats.
