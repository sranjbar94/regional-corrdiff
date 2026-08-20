#!/bin/bash
#SBATCH --job-name=fig_S9_v5
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_fig_S9_v5_%j.log
#SBATCH --account=pi_ey239

module load miniconda
conda activate corrdiff

# ── Paths ─────────────────────────────────────────────────────────────────────
BASEDIR=~/physicsnemo/examples/weather/corrdiff
SCRIPTS=$BASEDIR/regional_paper/scripts
OUTDIR=$BASEDIR/regional_paper/output
TRAIN_NC=~/DataPipeline_for_regional_hrrr_train_dataset/data/output/regional_hrrr_train_dataset_v5_train.nc

cd $BASEDIR/regional_paper
mkdir -p $OUTDIR

echo "============================================================"
echo "  Figure S9 (V5) — Training Data Coverage, full 100k samples"
echo "============================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

if [ ! -f "$TRAIN_NC" ]; then
    echo "ERROR: Missing $TRAIN_NC"
    exit 1
fi

python3 $SCRIPTS/figure_S9_training_data.py \
    --train $TRAIN_NC \
    --n-hist-samples 100000 \
    --outdir $OUTDIR

echo ""
echo "============================================================"
echo "  Figure S9 (V5) complete"
echo "============================================================"
echo "Output: $OUTDIR/figure_S9_training_data.pdf"
echo "Finished: $(date)"
