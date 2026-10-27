# Data format

All inputs are UTF-8 JSON Lines files supplied by the user. The repository does not contain query text, video frames, model weights, cached embeddings, or private annotations.

## Interval authority

The interval authority contains one record per query. Each record needs an `analysis.intervals_evaluated` integer. For an interval of length \(K\), the sequence length is recovered as `intervals_evaluated + K - 1`.

```json
{"query_id":"q-001","analysis":{"intervals_evaluated":85}}
```

## Method authority

Each method authority contains one successful result per query and must use the same query IDs as the interval authority.

```json
{"query_id":"q-001","status":"success","measurements":{"changed_frame_count":16,"psnr":55.2,"ssim":0.998,"uses_mock":false}}
```

`psnr` is the pooled full-sequence PSNR in decibels. Use `null` for infinite PSNR when the clean and adversarial tensors are identical. `ssim` is the uniformly frame-averaged full-sequence SSIM. Untouched frames must be unchanged. The report rejects missing queries, duplicate IDs, failed rows, mock rows, and inconsistent edited-frame counts.

## Selector inputs

The deterministic selectors accept ordered clean window scores, their contiguous frame positions, the full sequence length, and the target interval. Gradient-LSE accepts differentiable clean window scores as a function of normalized keyframe embeddings and uses the fixed temperature in `configs/paper.json`.
