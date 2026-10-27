# Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval

This repository provides a clean Python implementation of the interval-selection rules, temporally regularized PGD, and per-query fidelity reconstruction described in the paper.

## Methods

The three primary selectors use the same contiguous 16-keyframe budget:

| Selector | Rule |
|---|---|
| Target-centered | Edit the interval centered on the annotated target segment. |
| Highest-score window | Edit the interval aligned with the highest-scoring clean retrieval window. |
| Minimum-unaffected floor | Choose the interval that minimizes the highest clean score among unaffected windows. |

These rules have no learned weights or thresholds. Exact ties are resolved in favor of the earliest interval. The additional Gradient-LSE selector uses the paper's fixed temperature of 0.01.

`analyze_intervals` receives the clean score for each window and the zero-based target keyframe. It returns all three primary intervals and their unaffected-window diagnostics:

```python
from temporal_interval_selection import analyze_intervals, target_centered_positions

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
```

The PGD implementation uses 15 updates, an \(\ell_\infty\) budget of 8/255, a step size that decays from 2/255 to 10% of its initial value, and a smoothness weight of 0.5. Temporal regularization is normalized over the complete video sequence, including the zero perturbations outside the selected interval.

## Repository layout

```text
configs/
  paper.json                         Published attack and retrieval settings
docs/
  data-format.md                     Input authority formats
  reproduction.md                    Cohorts and reproduction procedure
scripts/
  rebuild_fidelity_report.py         Per-query PSNR/SSIM reconstruction
src/temporal_interval_selection/
  fidelity.py                        Whole-sequence to edited-frame conversion
  gradient_lse.py                    Gradient-LSE interval selection
  results.py                         Validated per-query aggregation
  scheduled_attack.py                PGD settings and memory-bounded attack
  selectors.py                       Three primary deterministic selectors
  temporal.py                        Full-sequence temporal penalties
```

## Installation

Python 3.10 or newer and a CUDA-enabled PyTorch installation are required for attack optimization.

```bash
python -m pip install .
```

## Rebuild edited-frame fidelity

The report command reads saved per-query authorities. It recovers edited-frame metrics for each query first, then computes cohort means.

```bash
tis-fidelity \
  --interval-authority data/intervals.jsonl \
  --authority target_centered=data/target_centered.jsonl \
  --authority highest_score_window=data/highest_score_window.jsonl \
  --authority minimum_unaffected_floor=data/minimum_unaffected_floor.jsonl \
  --output results/fidelity.json
```

Add another `--authority METHOD=FILE` option for an additional selector. Input fields and validation rules are described in [the data format](docs/data-format.md).

## Run an attack

The library function `run_scheduled_pgd` accepts one video tensor, one query embedding, the selected interval, and small adapter hooks for the retrieval model. Only the editable frames are held as trainable GPU state. The caller supplies the dataset loader and the frozen text-to-video model so that local data and model weights remain outside the source package.

```python
from temporal_interval_selection import PGDConfig, run_scheduled_pgd

adversarial_video, metrics = run_scheduled_pgd(
    runtime,
    adapter,
    video_frames,
    query_embedding,
    PGDConfig(),
    window_size=16,
    stride=1,
    changed_positions=selection.minimum_unaffected_floor.positions,
)
```

The input video tensor must have shape `(T, C, H, W)` with RGB values in `[0, 1]`. `runtime` provides sliding-window construction, affected-window lookup, and per-frame SSIM; `adapter` provides differentiable video embeddings. Keep the source video on CPU to retain the attack's bounded GPU memory use.

See [the reproduction guide](docs/reproduction.md) for the evaluation cohorts, required inputs, and run sequence.

## Data and provenance

The paper evaluates a 48-query competition cohort and a 100-query primary cohort. The 48-query cohort is a secondary descriptive evaluation; the primary cohort includes the additional Gradient-LSE comparison and the edited-frame fidelity table.

The video collection, query annotations, model weights, cached embeddings, and per-query attack authorities are not distributed here. Supply the authorized inputs through the documented interfaces. This repository implements the published equations and aggregation procedure; it does not claim byte-for-byte identity with a historical notebook runtime. Reproducing the reported numbers requires matching the original query mappings, model, preprocessing, and saved execution authorities.

The included temporal regularizer uses the full video length (T) in its denominators, as written in Equation (4). A preserved experiment runtime computes the temporal means over the affected crop instead; the available source files do not establish that this snapshot produced every reported result. Keep each reported result tied to its own runtime and per-query authority.

## Citation

If you use this code, cite the paper:

```text
P. N. L. Phan and N. H. Luong, “Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval,” SOICT 2026.
```
