#!/bin/bash
#SBATCH --job-name=v5_test
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --account=pi_ey239
#SBATCH --output=logs/v5_test_%j.log

module load miniconda
conda activate corrdiff_pipeline
cd ~/DataPipeline_for_regional_hrrr_train_dataset

python -W ignore run_pipeline_v5.py build \
    --config configs/pipeline_config_v5_test.yaml \
    --build_workers 10

DONE=$(python -c "
import json
ckpt = json.load(open('data/checkpoints_v5_test/build_checkpoint.json'))
print(ckpt['samples_done'])
")
echo "Samples done: $DONE / 3000"

if [ "$DONE" -lt 3000 ]; then
    echo "Build incomplete. Auto-resubmitting..."
    sbatch scripts/slurm_build_v5_test.sh
else
    echo "=== V5 TRAIN BUILD COMPLETE ==="
fi
