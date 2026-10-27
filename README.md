# Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval

This repository contains the code for the paper “Temporal Interval Selection for Adversarial Rank Suppression in Text-to-Video Retrieval,” accepted as a full paper at SOICT 2026.

## Setup

Python 3.10 or newer and CUDA-enabled PyTorch are required for attack optimization.

```bash
python -m pip install .
```

## Usage

`analyze_intervals` selects the target-centered, highest-score, and minimum-unaffected-floor intervals from clean retrieval scores. The package also includes the Gradient-LSE selector and scheduled PGD attack; both use adapters supplied by the caller.

Rebuild edited-frame PSNR and SSIM from per-query result files:

```bash
tis-fidelity --interval-authority intervals.jsonl --authority method=results.jsonl --output fidelity.json
```

See [the data format](docs/data-format.md) and [reproduction guide](docs/reproduction.md) for input details.

## Data

Datasets, model checkpoints, and saved experiment results are not included.