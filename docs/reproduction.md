# Reproduction guide

## What this repository can verify

After installation, run:

```bash
python -m unittest discover -s tests -v
```

These CPU smoke tests validate the paper-aligned formulas and invariants with a toy
adapter:

- full-sequence first- and second-order temporal normalization;
- zero initialization, PGD bounds, and selected-frame isolation;
- per-query PSNR/SSIM reconstruction before cohort averaging; and
- the frozen attack defaults in `configs/paper.json`.

They do not run the SigLIP model or verify any paper table.

## Paper cohorts

The camera-ready paper reports two separate cohorts:

- an earlier 48-query competition cohort, used for descriptive context; and
- a 100-query primary cohort on 100 distinct target videos.

The paper compares target-centered, highest-score-window, and minimum-unaffected-floor
on both cohorts. It also evaluates Gradient-LSE on the 100-query primary cohort. Keep
authorities from the two cohorts and four strategies separate.

## Required inputs for a full rerun

A numerical rerun requires all of the following external, versioned inputs:

1. locked query text and target-video/frame mappings for one cohort;
2. the original sampled keyframes and frame-index mapping;
3. the pinned SigLIP model checkpoint and exact preprocessing;
4. clean embeddings/scores for every stride-one window in the corpus;
5. a full-corpus reranking implementation; and
6. per-query authority storage for interval choices, ranks, and fidelity metrics.

The package accepts video tensors in `(T, C, H, W)` order with RGB values in `[0, 1]`.
The caller supplies the model adapter and `RunnerHooks` for sliding windows, affected
window lookup, and SSIM. CUDA is recommended for a real run; CPU is suitable only for
small functional checks.

## Run sequence

1. Compute clean scores for every stride-one window and map the reviewed target frame to
   a zero-based keyframe index.
2. Call `target_centered_positions` and `analyze_intervals` using the complete ordered
   window list. The selector rejects incomplete or non-stride-one lists.
3. Run `run_scheduled_pgd` for each selected interval with the same `PGDConfig`.
4. Re-encode every target window and rank the target against the complete 873-video
   corpus. The package does not implement this corpus-level stage.
5. Save one successful authority row per query and method.
6. Rebuild edited-frame fidelity with `tis-fidelity`; it validates matching query sets,
   recovers each query independently, and only then averages.

## Fidelity assumptions

The reconstruction command expects pooled full-sequence MSE-derived PSNR and uniformly
frame-averaged SSIM. Untouched frames are assumed unchanged. The algebra is valid only
under those definitions; it is not valid for crop-only metrics or cohort-averaged
metrics.

## Runtime and provenance boundary

The package follows the camera-ready protocol: full-video temporal normalization,
full-sequence fidelity, zero-start PGD, and a step size decaying to 10% of 2/255. A
historical runtime snapshot used during research instead computed temporal/image
statistics on an affected crop and used a constant step size. It is not established as
the exact source of every reported result. Do not combine its authorities with this
package's outputs.

The repository does not include the dataset, model weights, corpus cache, locked
annotations, or per-query execution authorities. Consequently, matching the reported
ranks and aggregate tables requires access to those artifacts in addition to this code.
