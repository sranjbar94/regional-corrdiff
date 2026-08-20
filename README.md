# Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas using Residual Corrective Diffusion Modeling

**Sadegh Ranjbar, Lucas Gloege, Noah Planavsky, Elizabeth Yankovsky**  
Department of Earth and Planetary Sciences, Yale University, New Haven, CT, USA  
Yale Center for Natural Carbon Capture, Yale University, New Haven, CT, USA

---

## Overview

This repository contains all code, configuration files, and figure-generation scripts associated with the paper:

> Ranjbar, S., Gloege, L., Planavsky, N., & Yankovsky, E. (2025). *Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas using Residual Corrective Diffusion Modeling.* [Journal]. [DOI]

We adapt **CorrDiff** — a residual corrective diffusion model — to downscale ERA5 atmospheric reanalysis (~25–28 km) to 3 km resolution over coastal regions of the contiguous United States (CONUS), using the High-Resolution Rapid Refresh (HRRR) dataset as the training reference. The model produces probabilistic, kilometer-scale atmospheric forcing for nine surface variables required by regional ocean models.

---

## Repository Structure

```
regional-corrdiff/
├── data_pipeline/          # V5 training dataset collection (ERA5 → HRRR patches)
├── training/               # Model training configs and sbatch scripts (G1, G2, G3)
├── inference/              # Inference configs and run scripts for each group
├── normalization/          # Per-group normalization statistics (stats.json)
├── figures/                # All paper figure generation scripts
├── LICENSE
└── README.md
```

---

## Model Architecture

The model follows the CorrDiff architecture ([Mardani et al., 2024](https://arxiv.org/abs/2309.15214)) and is implemented using the [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) framework (Apache 2.0).

Variables are split across three independent model groups to allow targeted training on physically distinct processes:

| Group | Variables | Description |
|-------|-----------|-------------|
| **G1** | 2-m temperature, 10-m U/V wind, surface pressure, specific humidity | Thermodynamic / momentum |
| **G2** | Total precipitation, snowfall | Precipitation |
| **G3** | Shortwave radiation, longwave radiation | Surface radiation |

Each group has a **regression** (deterministic) and a **diffusion** (stochastic) checkpoint.

---

## Data

### Training & Test Datasets
- **Training set**: 100,000 ERA5–HRRR paired ocean-patch samples, 2017–2023, full CONUS coastal coverage
- **Test set**: 3,000 samples, 2024–2025, independent evaluation period
- Datasets are archived on Yale Bouchet HPC project storage and are available upon reasonable request to the corresponding author (sadegh.ranjbar@yale.edu)

### ERA5 Input Data
Publicly accessible via the ARCO GCS zarr archive:
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803
- ARCO ERA5: https://cloud.google.com/storage/docs/public-datasets/era5

### HRRR Target Data
Publicly accessible via the NOAA AWS S3 archive (`noaa-hrrr-bdp-pds`):
- James, E. P., et al. (2022). The High-Resolution Rapid Refresh (HRRR): An hourly updating convection-allowing forecast model. *Bulletin of the American Meteorological Society*, 103(10), E2737–E2760. https://doi.org/10.1175/BAMS-D-21-0002.1
- HRRR S3: https://registry.opendata.aws/noaa-hrrr-pds/

---

## Model Weights

Pre-trained regression and diffusion checkpoints for all three groups (G1, G2, G3) are archived via Zenodo and linked in the paper. Due to file size (~500 MB per checkpoint), they are not hosted directly in this repository.

---

## Requirements

- Python ≥ 3.10
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (Apache 2.0)
- netCDF4, xarray, numpy, scipy, matplotlib, geopandas
- Access to ERA5 data (via ARCO GCS or local download) and HRRR data (via NOAA S3)
- HPC environment with SLURM scheduler (scripts written for Yale Bouchet cluster; adapt `#SBATCH` headers as needed)

Install dependencies:
```bash
conda create -n corrdiff python=3.10
conda activate corrdiff
pip install physicsnemo netCDF4 xarray numpy scipy matplotlib geopandas geodatasets
```

---

## Quick Start

### 1. Build the training dataset
```bash
cd data_pipeline
python run_pipeline_v5.py build --config configs/pipeline_config_v5_train.yaml
# or on HPC:
sbatch scripts/slurm_build_v5_train.sh
```

### 2. Train the models
```bash
cd training
# Example: G1 regression
sbatch scripts/run_train_g1_regression.sh
```

### 3. Run inference
```bash
cd inference
sbatch scripts/run_inference_g1.sh
```

### 4. Generate paper figures
```bash
cd figures
sbatch run_all_figures.sh
```

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{ranjbar2025corrdiff,
  title   = {Kilometer-Scale Atmospheric Downscaling for Coastal Ocean Areas
             using Residual Corrective Diffusion Modeling},
  author  = {Ranjbar, Sadegh and Gloege, Lucas and Planavsky, Noah and Yankovsky, Elizabeth},
  journal = {[Journal]},
  year    = {2025},
  doi     = {[DOI]}
}
```

---

## License

This repository is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

The NVIDIA PhysicsNeMo framework is also distributed under the Apache 2.0 license.
ERA5 data are subject to the Copernicus Climate Change Service (C3S) Terms of Use.
HRRR data are in the public domain (U.S. Government work).

---

## Contact

Sadegh Ranjbar — sadegh.ranjbar@yale.edu  
Department of Earth and Planetary Sciences, Yale University
