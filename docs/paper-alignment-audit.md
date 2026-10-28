# Paper alignment audit

This audit compares the source package with the attached camera-ready manuscript. The paper
source/PDF is not included in this repository, so the paper-side details were checked
against the manuscript artifact available in the surrounding research workspace. This
is a static/code audit; it is not a fresh 48- or 100-query experiment.

## Alignment matrix

| Paper requirement | Source | Status |
| --- | --- | --- |
| 16 consecutive editable keyframes | `configs/paper.json`, `selectors.py` | Implemented |
| Stride-one temporal windows | `selectors.py` | Implemented; incomplete/non-stride-one inputs are rejected |
| Target-centered interval: seven before, eight after when possible | `target_centered_positions` | Implemented |
| Highest-score-window baseline with earliest tie | `analyze_intervals` | Implemented |
| Minimum unaffected-floor selector over every valid start | `analyze_intervals` | Implemented |
| Gradient-LSE selector, τ = 0.01 | `gradient_lse.py` | Implemented |
| Zero-start PGD | `run_scheduled_pgd` | Implemented |
| ε = 8/255, first step = 2/255, 15 updates | `PGDConfig` | Implemented |
| Linear decay to 10% of initial step | `run_scheduled_pgd` | Implemented |
| First-order smoothness, λ₁ = 0.5 | `temporal.py`, `scheduled_attack.py` | Implemented |
| Temporal normalization over all T frames | `temporal_regularizers` | Implemented and smoke-tested |
| Full-sequence PSNR/SSIM reconstruction | `fidelity.py`, `results.py` | Implemented under documented metric assumptions |
| Full-corpus target reranking | Caller responsibility | Not implemented in this package |
| SigLIP model, preprocessing, 873-video corpus | External artifacts | Not included |
| Reported 48/100-query authorities and aggregate tables | External artifacts | Not included |
| SigLIP query construction and 873-video full-corpus reranking | Caller/runtime | Not implemented in this package |

## Reproduction conclusion

The algorithmic core is consistent with the camera-ready protocol. The repository can therefore
reproduce selector decisions and run a small attack when a compatible adapter is supplied.
It cannot, from the checked-in files alone, reproduce the paper's final ranks or aggregate
numbers because the data, checkpoint, corpus cache, locked annotations, and authorities
are external.

The historical runtime used in the research process also differs from this package in
crop-only temporal/image normalization and constant rather than decaying PGD steps.
Those outputs must remain provenance-separated from this implementation.

## Checks run

From the package directory:

```text
python -m unittest discover -s tests -v
Ran 5 tests ... OK
```

The test suite uses a toy differentiable adapter and synthetic tensors. Passing it is
evidence for local semantics only; it is not evidence that the paper's empirical results
have been independently replayed.
