# Data Pipeline — ERA5 → HRRR Ocean-Patch Dataset (V5)

This module builds the paired ERA5–HRRR training and test datasets used in the paper. It extracts co-located ocean-dominated 32×32 ERA5 patches (≈8°×8°, ~25 km/pixel) and their corresponding 256×256 HRRR patches (≈768 km×768 km, ~3 km/pixel) from CONUS coastal ocean locations.

## V5 Dataset Summary

| Split | Samples | Period | Coverage |
|-------|---------|--------|----------|
| Train | 100,000 | 2017–2023 | Full CONUS coastal ocean |
| Test  | 3,000   | 2024–2025 | Full CONUS coastal ocean |

**Geographic domain**: 22–50°N, 127–66°W (Pacific + Gulf of Mexico + Atlantic coasts)  
**Ocean fraction threshold**: ≥60% of the 32×32 ERA5 patch must be ocean (LSM < 0.2)

## Key Fixes vs. Earlier Versions (V2–V4)

- **Domain bounds bug (V2–V4)**: The `lon_min=-120` + 4° patch-buffer cutoff excluded the Pacific coast entirely and most of the Gulf. V5 uses `lon_min=-127`, `lat_min=22` to correctly cover all three coasts.
- **Coordinate metadata bug (V2–V4)**: Earlier files stored ERA5-grid integer indices in `coord` but their `invariant/latitude` and `invariant/longitude` groups held a different, unrelated HRRR reference grid — making any downstream lat/lon lookup produce physically meaningless values. V5's `nc_writer_v5.py` adds real `lat`/`lon` float arrays directly to the root group.

## Usage

### Build training set
```bash
python run_pipeline_v5.py build --config configs/pipeline_config_v5_train.yaml
# on HPC:
sbatch scripts/slurm_build_v5_train.sh
```

### Build test set
```bash
python run_pipeline_v5.py build --config configs/pipeline_config_v5_test.yaml
# on HPC:
sbatch scripts/slurm_build_v5_test.sh
```

## ERA5 Data Requirements

The pipeline reads ERA5 single-level and pressure-level fields from local NetCDF files. Download ERA5 data via the CDS API or the ARCO GCS zarr archive before running. Set the `era5_raw_dir` path in the config accordingly.

## Output Schema (V5 NetCDF)

```
root variables:
  time   (sample,)          UTC timestamp (seconds since 1970-01-01)
  coord  (sample, 2)        ERA5-grid [lat_idx, lon_idx] — kept for backward compat.
  lat    (sample,)          REAL patch-centre latitude (°N)  ← use this for mapping
  lon    (sample,)          REAL patch-centre longitude (°E, -180 to 180)

groups:
  input/   ERA5 variables at 32×32 resolution (u10, v10, t2m, sp, q, ssrd, strd, tp, sf, ...)
  output/  HRRR variables at 256×256 resolution (10u, 10v, 2t, sp, q, ssrd, strd, tp, sf)
  invariant/  HRRR-grid reference (latitude, longitude, elev_mean, lsm_mean) — 1059×1799
```
