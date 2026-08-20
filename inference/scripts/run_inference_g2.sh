#!/bin/bash
#SBATCH --job-name=eval_g2cd_16ens
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=h200:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --account=pi_ey239
#SBATCH --output=slurm_g2_cd_eval_16ens_%j.log

# ──────────────────────────────────────────────────────────────
# G2 Precipitation (CorrDiff) eval — 16 ensemble members
# Variables: tp, sf
# Test set:  data/v4/g2/training_g2_corrdiff.nc — first 3,000 samples
# Chunks:    60 chunks of 50 samples each (60 × 50 = 3,000)
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd ~/physicsnemo/examples/weather/corrdiff
module load miniconda && conda activate corrdiff

# ── Checkpoints ────────────────────────────────────────────────
REG=$(ls -t checkpoints_g2_corrdiff_regression/CorrDiffRegressionUNet.*.mdlus 2>/dev/null | head -1)
DIFF=$(ls -t checkpoints_g2_corrdiff_diffusion/EDMPrecondSuperResolution.*.mdlus 2>/dev/null | head -1)

if [[ -z "$REG" || -z "$DIFF" ]]; then
    echo "ERROR: G2 CorrDiff checkpoints not found. Check:"
    echo "  checkpoints_g2_corrdiff_regression/"
    echo "  checkpoints_g2_corrdiff_diffusion/"
    exit 1
fi
echo "G2 regression checkpoint: $REG"
echo "G2 diffusion checkpoint:  $DIFF"

# ── Output dir ─────────────────────────────────────────────────
OUTDIR=outputs/figure_data_g2_corrdiff
mkdir -p "$OUTDIR"
CHUNK_DIR="$OUTDIR/chunks"
mkdir -p "$CHUNK_DIR"

N_SAMPLES=3000   # use first 3k test samples (dataset has 8,200 available)
N_CHUNKS=60      # 3000 / 50 = 60 chunks
CHUNK_SIZE=50
N_ENSEMBLES=16

echo "Starting G2 CorrDiff inference: $N_SAMPLES samples, $N_CHUNKS chunks, $N_ENSEMBLES ensemble members"

# ── Run inference in chunks ─────────────────────────────────────
for i in $(seq 0 $((N_CHUNKS - 1))); do
    CHUNK_FILE="$CHUNK_DIR/chunk_${i}.nc"
    if [[ -f "$CHUNK_FILE" ]]; then
        echo "Chunk $i already exists — skipping"
        continue
    fi
    OFFSET=$((i * CHUNK_SIZE))
    torchrun --standalone --nproc_per_node=2 generate.py \
        --config-name=config_generate_g2_corrdiff.yaml \
        ++generation.io.reg_ckpt_filename="$REG" \
        ++generation.io.res_ckpt_filename="$DIFF" \
        ++generation.num_ensembles=$N_ENSEMBLES \
        ++generation.io.output_filename="$CHUNK_FILE" \
        ++generation.io.sample_offset=$OFFSET \
        ++generation.io.n_samples=$CHUNK_SIZE \
        ++dataset.data_path=data/v4/g2/training_g2_corrdiff.nc \
        ++dataset.stats_path=data/v4/g2/stats_g2_corrdiff.json
    echo "Chunk $i done → $CHUNK_FILE"
done

echo "All G2 CorrDiff chunks complete. Running scorer..."

# ── Score ───────────────────────────────────────────────────────
python score_g2_corrdiff.py \
    --chunk_dir "$CHUNK_DIR" \
    --output_dir "$OUTDIR" \
    --output_prefix eval_g2_corrdiff \
    --n_ensembles $N_ENSEMBLES \
    --delete_chunks

echo "G2 CorrDiff scoring complete. Output in $OUTDIR/"
ls -lh "$OUTDIR/"
