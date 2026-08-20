# Inference

Generates 16-member ensemble predictions using PhysicsNeMo's generate.py.
See: https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff

## Run
```bash
sbatch scripts/run_inference_g1.sh   # G1: thermodynamic variables
sbatch scripts/run_inference_g2.sh   # G2: precipitation
sbatch scripts/run_inference_g3.sh   # G3: radiation
```

## Output
Chunk NetCDF files in outputs/figure_data_g{1,2,3}/chunk_*.nc, merged by
score_g{1,2,3}.py into evaluation files used by the figure scripts.
