# Training

Uses NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) framework.

## Two-stage training per group
1. **Regression**: deterministic UNet predicting conditional mean
2. **Diffusion**: stochastic CorrDiff trained on residuals

## Run (example G1)
```bash
# From physicsnemo/examples/weather/corrdiff/
python train.py --config-name=config_training_g1_thermo_regression.yaml
python train.py --config-name=config_training_g1_thermo_diffusion.yaml
```

Pre-trained checkpoints are archived on Zenodo (see paper for DOI).
