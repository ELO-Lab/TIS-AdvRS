# Reproduction guide

## Evaluation cohorts

The paper reports results for two distinct query cohorts:

- The competition cohort contains 48 queries and is reported descriptively for the three primary interval selectors.
- The primary cohort contains 100 queries. It includes the additional Gradient-LSE comparison and the edited-frame fidelity table.

Keep cohort annotations and saved per-query authorities separate. The same selector names or query count do not establish that two authority files refer to the same cohort.

## Required inputs

To run an attack, provide:

1. The authorized query catalog and target video/frame mappings for the selected cohort.
2. The original video keyframes and frame-index mapping.
3. The pinned text-to-video model and preprocessing.
4. The complete corpus embeddings or an equivalent validated retrieval index.
5. A `RunnerHooks` implementation for sliding-window construction, affected-window lookup, and SSIM.

The attack API accepts tensors in `(T, C, H, W)` order with RGB values in `[0, 1]`. Keep the input video on CPU; the attack moves only the selected frames and one retrieval window at a time to the configured CUDA device. The video encoder adapter must return the normalized video embedding used by the retrieval model. The query embedding and retrieval settings must match the run protocol.

## Run sequence

1. Compute clean retrieval scores for every temporal window and map the annotated target keyframe to a zero-based index.
2. Call `target_centered_positions` and `analyze_intervals` to obtain the three primary intervals from the same clean scores and fixed interval length.
3. Run `run_scheduled_pgd` with the selected positions and the shared attack settings.
4. Save one successful authority row per query and method, including full-sequence PSNR/SSIM and the selected frame positions.
5. Rebuild edited-frame fidelity with `tis-fidelity`. The command validates matching query sets, converts each query independently, and only then averages.

## Fidelity assumptions

For PSNR, sequence MSE is pooled over every pixel in the video. For SSIM, per-frame SSIM is averaged uniformly across the sequence. Untouched frames contribute zero MSE and SSIM equal to one. The algebraic recovery is valid under these definitions; it is not valid for crop-only metrics or cohort-averaged metrics.

The result reconstruction command does not rerun attacks or alter saved authorities. Numerical reproduction also depends on matching the data, annotations, model checkpoint, preprocessing, and runtime used for the experiment.

## Implementation scope

This package normalizes temporal losses over the full video. A preserved experiment runtime computes the means over the crop containing the affected retrieval windows, so results from the two implementations are not numerically interchangeable. Match each result to its runtime and per-query authority.
