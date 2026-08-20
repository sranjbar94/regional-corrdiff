#!/bin/bash
#SBATCH --job-name=g1_eval_16ens
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_g1_eval_16ens_%j.log
#SBATCH --account=pi_ey239
module load miniconda
conda activate corrdiff
cd ~/physicsnemo/examples/weather/corrdiff
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "============================================================"
echo "G1 Thermo — Evaluation (2t, 10u, 10v, sp, q)"
echo "16 ensemble members, 3k samples"
echo "============================================================"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
REG_CKPT=$(ls -t checkpoints_g1_regression/CorrDiffRegressionUNet.*.mdlus 2>/dev/null | head -1)
DIFF_CKPT=$(ls -t checkpoints_g1_diffusion/EDMPrecondSuperResolution.*.mdlus 2>/dev/null | head -1)
echo "Regression:  $REG_CKPT"
echo "Diffusion:   $DIFF_CKPT"
if [ -z "$REG_CKPT" ] || [ -z "$DIFF_CKPT" ]; then
    echo "ERROR: Missing checkpoint(s)!"
    exit 1
fi
OUTDIR=outputs/figure_data_g1
mkdir -p $OUTDIR
mapfile -t ALL_TIMES < <(python3 -c "
import json
with open('outputs/eval_times_3k.json') as f:
    times = json.load(f)
for t in times:
    print(t)
")
TOTAL=${#ALL_TIMES[@]}
echo "Total timestamps: $TOTAL"
CHUNK_SIZE=50
N_CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo ">>> G1 CorrDiff evaluation: $TOTAL timestamps x 16 ensemble"
echo ">>> $N_CHUNKS chunks of $CHUNK_SIZE"
for (( c=0; c<N_CHUNKS; c++ )); do
    OUTFILE="$OUTDIR/chunk_${c}.nc"
    if [ -f "$OUTFILE" ]; then echo "  Chunk $c: EXISTS, skipping"; continue; fi
    START=$(( c * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    TIMES=""
    for (( i=START; i<END; i++ )); do
        if [ -z "$TIMES" ]; then TIMES="\"${ALL_TIMES[$i]}\""
        else TIMES="$TIMES,\"${ALL_TIMES[$i]}\""; fi
    done
    echo "  Chunk $c ($START-$((END-1))): generating..."
    torchrun --standalone --nproc_per_node=2 generate.py \
        --config-name=config_generate_g1.yaml \
        ++generation.io.res_ckpt_filename=$DIFF_CKPT \
        ++generation.io.reg_ckpt_filename=$REG_CKPT \
        ++generation.num_ensembles=16 \
        ++generation.io.output_filename=$OUTFILE \
        ++generation.times="[$TIMES]" \
        ++dataset.data_path=./data/v4/randomly/test.nc \
        ++dataset.stats_path=./data/v4/randomly/stats.json
    if [ $? -eq 0 ]; then echo "  Chunk $c: DONE ($(date))"
    else echo "  Chunk $c: FAILED ($(date))"; fi
done
echo ">>> Evaluation complete at $(date)"
echo ">>> Chunks: $(ls $OUTDIR/chunk_*.nc 2>/dev/null | wc -l) / $N_CHUNKS"
