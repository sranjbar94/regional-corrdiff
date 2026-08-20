# Data Pipeline

Builds the V5 paired ERA5-HRRR training and test datasets.

## Usage
```bash
# Training set (100,000 samples, 2017-2023)
sbatch scripts/slurm_build_v5_train.sh

# Test set (3,000 samples, 2024-2025)
sbatch scripts/slurm_build_v5_test.sh
```

## Key V5 Fixes vs V2-V4
- Domain bounds now cover full CONUS coastal ocean (Pacific + Gulf + Atlantic)
- Real lat/lon float arrays stored directly in root group (no more coord/invariant grid mismatch)
- ERA5 tp/sf unit conversion applied (m → mm/hr)

## Output Schema

root: time, coord, lat, lon
groups: input/ (ERA5, 32×32), output/ (HRRR, 256×256), invariant/

