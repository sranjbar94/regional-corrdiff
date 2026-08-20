# Model Training — G1, G2, G3

Training uses the [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) framework. Each variable group has two training stages: a **regression** (deterministic UNet) followed by a **diffusion** (stochastic CorrDiff) stage that uses the regression output as its mean prior.

## Groups

| Group | Variables | Training data |
|-------|-----------|---------------|
| G1 | 2t, 10u, 10v, sp, q | V5 train (100,000 samples) |
| G2 | tp, sf | V5 train (100,000 samples) |
| G3 | ssrd, strd | V5 train (100,000 samples) |

## Training Stages

### Stage 1 — Regression (deterministic UNet)
Trains a deterministic UNet to predict the conditional mean of the HRRR target given the ERA5 input. This provides a strong prior for the diffusion stage.

```bash
# G1 example
sbatch scripts/run_train_g1_regression.sh
```

### Stage 2 — Diffusion (residual CorrDiff)
Trains the stochastic diffusion model on the residuals between HRRR target and regression output. Requires the regression checkpoint from Stage 1.

```bash
# G1 example
sbatch scripts/run_train_g1_diffusion.sh
```

## Configuration

Each group's YAML config (in `configs/`) specifies:
- `data_path`: path to the V5 training NetCDF
- `regression_checkpoint`: path to Stage 1 checkpoint (required for Stage 2)
- Network architecture, learning rate, batch size, number of steps

## Notes on the PhysicsNeMo Framework

Training is launched via PhysicsNeMo's `train.py` entry point. Refer to the [PhysicsNeMo documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) for full details on training configuration options, multi-GPU setup, and checkpoint management.

## Hardware

Training was performed on Yale Bouchet HPC cluster using NVIDIA H200 GPUs. Approximate wall-clock times per group:
- Regression: ~12–24 hours on 1× H200
- Diffusion: ~48–72 hours on 1× H200
