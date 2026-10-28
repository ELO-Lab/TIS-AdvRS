# Temporal Interval Selection for Adversarial Rank Suppression

Reference implementation for **“Interval Selection for Adversarial Rank Suppression in
Windowed Text-to-Video Retrieval”** (SOICT 2026). The package contains the deterministic
interval selectors, the paper-aligned scheduled PGD core, and per-query fidelity
reconstruction.

## What is implemented

The three selectors use one fixed budget of 16 consecutive keyframes and stride-one
retrieval windows:

| Selector | Rule |
| --- | --- |
| `target_centered` | Center the interval on the reviewed target frame. |
| `highest_score_window` | Use the clean window with the highest query score. |
| `minimum_unaffected_floor` | Choose the interval minimizing the highest clean score among windows it cannot affect. |

The selector has no fitted weights or thresholds. Ties use the earliest valid start.
The attack core follows the paper's shared optimizer: zero initialization, 15 PGD
updates, an ℓ∞ bound of 8/255, a step size decaying from 2/255 to 10% of
that value, and first-order temporal smoothness with weight 0.5. Temporal losses and
sequence-level fidelity are normalized over the complete video, including unchanged
frames outside the selected interval.

`gradient_lse.py` is retained as an experimental utility from an earlier protocol. It
is not one of the three selectors reported in the current paper.

## Repository structure

```text
configs/
  paper.json                         Frozen window, budget, PGD, and utility settings
docs/
  data-format.md                     JSONL authority formats and metric assumptions
  reproduction.md                    Cohorts, required assets, and run order
scripts/
  rebuild_fidelity_report.py         CLI wrapper for per-query PSNR/SSIM recovery
src/temporal_interval_selection/
  selectors.py                       Three deterministic interval selectors
  scheduled_attack.py                Memory-bounded PGD and run metrics
  temporal.py                        Full-sequence first/second-order penalties
  fidelity.py                        Sequence-to-edited-frame metric conversion
  results.py                          Authority validation and aggregation
  gradient_lse.py                    Experimental earlier selector
tests/
  test_cpu_smoke.py                  Dependency-light semantic smoke tests
```

## Reproducibility status

The implementation is **paper-aligned at the algorithm level**, but this repository is
not a self-contained rerun of the reported experiment. The private/external inputs are
not included:

- the 873-video collection and sampled keyframes;
- locked query, target-video, and target-frame annotations for the 48-query earlier
  cohort and the 93-query prospective cohort;
- the pinned SigLIP checkpoint, preprocessing runtime, and complete corpus cache; and
- the per-query execution authorities used to produce the paper tables.

Therefore the included tests can validate selector semantics, temporal normalization,
PGD bounds, and fidelity arithmetic, but they cannot reproduce the paper's ranks,
Recall@10, ASR@10, PSNR/SSIM means, or significance tests. The attack function also
does not provide a dataset loader or full-corpus reranker; callers must supply the
model adapter, retrieval hooks, annotations, and authority writer.

The historical runtime snapshot used during research differs in some details (crop-only
normalization/metrics and a constant step size). It must not be mixed with this
paper-aligned package when interpreting numerical results. See
[the reproduction guide](docs/reproduction.md) for the exact boundary.
The point-by-point review is in [paper-alignment-audit.md](docs/paper-alignment-audit.md).

## Install

Python 3.10 or newer is required. PyTorch is needed for the attack API; CUDA is
recommended for realistic model runs, while the included smoke tests run on CPU.

```bash
python -m pip install .
```

Run the local checks after installation:

```bash
python -m unittest discover -s tests -v
```

## Use the library

```python
from temporal_interval_selection import (
    PGDConfig,
    analyze_intervals,
    run_scheduled_pgd,
    target_centered_positions,
)

target_interval = target_centered_positions(
    frame_count=len(video_frames),
    target_position=target_frame_index,
    window_size=16,
)
selection = analyze_intervals(
    clean_window_scores,
    windows,
    frame_count=len(video_frames),
    target_positions=target_interval,
    window_size=16,
)

adversarial_video, metrics = run_scheduled_pgd(
    runtime, adapter, video_frames, query_embedding, PGDConfig(),
    window_size=16, stride=1,
    changed_positions=selection.minimum_unaffected_floor.positions,
)
```

Inputs use `(T, C, H, W)` RGB tensors in `[0, 1]`. See
[data-format.md](docs/data-format.md) and [reproduction.md](docs/reproduction.md) for
authority schemas, metric assumptions, and the full evaluation sequence.

## Fidelity reconstruction

The CLI reads saved full-sequence metrics, recovers edited-frame PSNR/SSIM per query,
then averages the recovered values:

```bash
tis-fidelity \
  --interval-authority data/intervals.jsonl \
  --authority minimum_unaffected_floor=data/minimum_unaffected_floor.jsonl \
  --output results/fidelity.json
```

This command only validates and aggregates saved authorities; it does not rerun attacks.

## Citation

P. N. L. Phan and N. H. Luong, “Interval Selection for Adversarial Rank Suppression in
Windowed Text-to-Video Retrieval,” SOICT 2026.
