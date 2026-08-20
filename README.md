# Regional CorrDiff — Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas

**Sadegh Ranjbar, Lucas Gloege, Noah Planavsky, Elizabeth Yankovsky**

Department of Earth and Planetary Sciences, Yale University
Yale Center for Natural Carbon Capture, Yale University

## Overview

Code and configuration files for: *Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas using Residual Corrective Diffusion Modeling* (Ranjbar et al., 2025).

We adapt CorrDiff — a residual corrective diffusion model — to downscale ERA5 (~25 km) to 3 km resolution over CONUS coastal ocean regions using HRRR as the training reference. The model produces probabilistic atmospheric forcing for 9 surface variables required by regional ocean models.

## Repository Structure

regional-corrdiff/
├── data_pipeline/ # V5 training dataset collection (ERA5 → HRRR patches)
├── training/ # Model training configs for G1, G2, G3
├── inference/ # Inference configs and run scripts
├── normalization/ # Per-group normalization statistics
├── figures/ # All paper figure generation scripts
├── LICENSE
└── README.md


## Model Groups

| Group | Variables |
|-------|-----------|
| G1 | 2-m temperature, 10-m U/V wind, surface pressure, specific humidity |
| G2 | Total precipitation, snowfall |
| G3 | Shortwave radiation, longwave radiation |

Each group has a regression (deterministic) and diffusion (stochastic) checkpoint, archived on Zenodo (see paper for DOI).

## Data

- **Training set**: 100,000 ERA5-HRRR paired ocean-patch samples, 2017-2023, full CONUS coastal coverage
- **Test set**: 3,000 samples, 2024-2025
- Available upon reasonable request to the corresponding author (sadegh.ranjbar@yale.edu)
- ERA5: https://cloud.google.com/storage/docs/public-datasets/era5 (Hersbach et al., 2020)
- HRRR: https://registry.opendata.aws/noaa-hrrr-pds/ (James et al., 2022)

## Requirements

- Python >= 3.10, NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo)
- netCDF4, xarray, numpy, scipy, matplotlib, geopandas

## Quick Start

```bash
# 1. Build dataset
cd data_pipeline && sbatch scripts/slurm_build_v5_train.sh

# 2. Train (example: G1 regression)
cd training && sbatch scripts/run_train_g1_regression.sh

# 3. Generate figures
cd figures && sbatch run_all_figures.sh
```

## Citation

```
bibtex
@article{ranjbar2025corrdiff,
  title   = {Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas
             using Residual Corrective Diffusion Modeling},
  author  = {Ranjbar, Sadegh and Gloege, Lucas and Planavsky, Noah and Yankovsky, Elizabeth},
  year    = {2025}
}
```

## License

Apache License 2.0 — see LICENSE file.
