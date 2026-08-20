#!/bin/bash
#SBATCH --job-name=v5_train
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --account=pi_ey239
#SBATCH --output=logs/v5_train_%j.log

module load miniconda
conda activate corrdiff_pipeline
cd ~/DataPipeline_for_regional_hrrr_train_dataset

python -W ignore run_pipeline_v5.py build \
    --config configs/pipeline_config_v5_train.yaml \
    --build_workers 10

DONE=$(python -c "
import json
ckpt = json.load(open('data/checkpoints_v5_train/build_checkpoint.json'))
print(ckpt['samples_done'])
")
echo "Samples done: $DONE / 100000"

if [ "$DONE" -lt 100000 ]; then
    echo "Build incomplete. Auto-resubmitting..."
    sbatch scripts/slurm_build_v5_train.sh
else
    echo "=== V5 TRAIN BUILD COMPLETE ==="
fi
