#!/bin/bash
#SBATCH --job-name=fig_S5_ens
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_fig_S5_%j.log
#SBATCH --account=pi_ey239

module load miniconda
conda activate corrdiff

# ── Paths ─────────────────────────────────────────────────────────────────────
BASEDIR=~/physicsnemo/examples/weather/corrdiff
SCRIPTS=$BASEDIR/regional_paper/scripts
OUTDIR=$BASEDIR/regional_paper/output

G1_CHUNKS=$BASEDIR/outputs/figure_data_g1
G2_CHUNKS=$BASEDIR/outputs/figure_data_g2_corrdiff

cd $BASEDIR/regional_paper
mkdir -p $OUTDIR

echo "============================================================"
echo "  Figure S5 — CRPS Sensitivity to Ensemble Size (SLOW, separate)"
echo "============================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""
echo "Reading raw 16-member chunk files directly -- this is much slower"
echo "than the rest of the figure pipeline (run_all_figures.sh). Expect"
echo "several minutes to ~1 hour depending on --n-chunks/--n-subsets."
echo ""

# Full run: all 60 chunks per variable, 20 random subsets per (chunk, N<16).
# Reduce --n-chunks for a quick sanity check before committing to the full run.
python3 $SCRIPTS/figure_S5_ensemble_sensitivity.py \
    --g1-chunks $G1_CHUNKS \
    --g2-chunks $G2_CHUNKS \
    --n-chunks 60 \
    --n-subsets 20 \
    --pixel-stride 4 \
    --outdir $OUTDIR \
    --cache $OUTDIR/figure_S5_cache.json

echo ""
echo "============================================================"
echo "  Figure S5 complete"
echo "============================================================"
echo "Output: $OUTDIR/figure_S5_ensemble_sensitivity.pdf"
echo "Cache (for fast re-plotting): $OUTDIR/figure_S5_cache.json"
echo ""
echo "To re-plot with style changes WITHOUT re-reading chunks:"
echo "  python3 \$SCRIPTS/figure_S5_ensemble_sensitivity.py \\"
echo "      --from-cache $OUTDIR/figure_S5_cache.json --outdir $OUTDIR"
echo "Finished: $(date)"
