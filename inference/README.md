# Inference — Generating Ensemble Predictions

Inference uses the `generate.py` script from the [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) framework. This repository provides the group-specific configuration files and SLURM wrapper scripts; for a full explanation of the inference pipeline, argument options, and output format, refer to the [PhysicsNeMo CorrDiff example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff).

## Pre-trained Checkpoints

Regression and diffusion checkpoints for G1, G2, and G3 are archived on Zenodo (see paper for DOI). Download and place them as specified in each group's config file before running inference.

## Running Inference

Each group's inference script reads from the V5 test set and writes 16-member ensemble predictions to chunk NetCDF files used for evaluation and figure generation.

```bash
# G1
sbatch scripts/run_inference_g1.sh

# G2
sbatch scripts/run_inference_g2.sh

# G3
sbatch scripts/run_inference_g3.sh
```

## Output Format

Inference produces chunk files (one per timestamp batch) in `outputs/figure_data_g{1,2,3}/chunk_*.nc`, each with the schema:

```
groups:
  truth/        HRRR ground truth  (variables × H × W)
  prediction/   CorrDiff ensemble  (16 members × variables × H × W)
  input/        ERA5 bilinear      (variables × H × W)
```

These chunk files are subsequently merged by the scoring scripts (`score_g{1,2,3}.py`) into the evaluation NetCDF files used by the figure scripts.

## Normalization Statistics

The `../normalization/` directory contains `stats_g{1,2,3}.json` — per-variable mean and standard deviation computed over the training set. These are required by PhysicsNeMo's `generate.py` at inference time.
