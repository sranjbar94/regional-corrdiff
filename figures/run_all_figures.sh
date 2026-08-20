#!/bin/bash
#SBATCH --job-name=paper_figs
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_paper_figs_%j.log
#SBATCH --account=pi_ey239

module load miniconda
conda activate corrdiff

# ── Paths ─────────────────────────────────────────────────────────────────────
BASEDIR=~/physicsnemo/examples/weather/corrdiff
SCRIPTS=$BASEDIR/regional_paper/scripts
OUTDIR=$BASEDIR/regional_paper/output

# Eval data paths (on HPC)
G1=$BASEDIR/outputs/figure_data_g1/eval_g1_main.nc
G2=$BASEDIR/outputs/figure_data_g2_corrdiff/eval_g2_corrdiff_main.nc
G3=$BASEDIR/outputs/figure_data_g3/eval_g3_main.nc
G1S=$BASEDIR/outputs/figure_data_g1/eval_g1_stats.json
G2S=$BASEDIR/outputs/figure_data_g2_corrdiff/eval_g2_corrdiff_stats.json
G3S=$BASEDIR/outputs/figure_data_g3/eval_g3_stats.json
G1_LATLON=$BASEDIR/outputs/figure_data_g1/eval_g1_latlon.json
G2_LATLON=$BASEDIR/outputs/figure_data_g2_corrdiff/eval_g2_corrdiff_latlon.json
G3_LATLON=$BASEDIR/outputs/figure_data_g3/eval_g3_latlon.json

cd $BASEDIR/regional_paper
mkdir -p $OUTDIR

echo "============================================================"
echo "  Regional Paper — Figures & Tables"
echo "============================================================"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# ── Verify eval files exist ───────────────────────────────────────────────────
for f in $G1 $G2 $G3 $G1S $G2S $G3S; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done
echo "All 6 eval files found."
echo ""

# ── Figure 1 — Spatial comparison (one PDF per variable) ─────────────────────
echo ">>> Figure 1 (spatial) ..."
python3 $SCRIPTS/figure_01_spatial.py \
    --g1 $G1 --g2 $G2 --g3 $G3 \
    --outdir $OUTDIR
echo ""

# ── Figures 2-7 — Main results (skip 5, runs standalone below with full RAPSD) ─
# NOTE: Figure 4 (spread-skill) now covers all 9 variables (G1+G2+G3), not
# just the 5 G1 variables -- requires --g2/--g3 which were already passed.
echo ">>> Figures 2-7 (main) ..."
python3 $SCRIPTS/figures_main.py \
    --g1 $G1 --g2 $G2 --g3 $G3 \
    --g1s $G1S --g2s $G2S --g3s $G3S \
    --storms 1991 2507 2172 --skip 5 \
    --outdir $OUTDIR
echo ""

# ── Figure 5 standalone (corrected RAPSD, all 9 variables: 2t,10u,10v,sp,q,tp,sf,ssrd,strd) ─
echo ">>> Figure 5 standalone (corrected RAPSD, 9 variables) ..."
python3 $SCRIPTS/figure_05_rapsd.py \
    --g1 $G1 --g2 $G2 --g3 $G3 \
    --outdir $OUTDIR
echo ""

# ── Cold-front case study — 3 cases (a/b/c), with UNet column ───────────────
echo ">>> Cold-front figure (3 cases) ..."
python3 $SCRIPTS/figure_coldfront.py \
    --g1 $G1 \
    --idxs 1991 2507 2172 --angles 2 0 0 \
    --outdir $OUTDIR
echo ""

# ── Supplementary S1-S4 ─────────────────────────────────────────────────────
echo ">>> Supplementary figures (S1-S4) ..."
python3 $SCRIPTS/figures_supp.py \
    --g1 $G1 --g2 $G2 --g3 $G3 \
    --g1-latlon $G1_LATLON --g2-latlon $G2_LATLON \
    --outdir $OUTDIR
echo ""

# ── Tables 1-3 + S1 ─────────────────────────────────────────────────────────
echo ">>> Tables ..."
python3 $SCRIPTS/tables.py \
    --g1 $G1 --g2 $G2 --g3 $G3 \
    --g1s $G1S --g2s $G2S --g3s $G3S \
    --outdir $OUTDIR
echo ""

echo "============================================================"
echo "  All figures and tables complete"
echo "============================================================"
echo "Output: $OUTDIR"
echo "NOTE: Two figures are NOT included here and run as separate scripts:"
echo "  Figure S5 (CRPS vs ensemble size) -- reads raw 16-member chunks:"
echo "        sbatch $SCRIPTS/run_figure_S5.sh"
echo "  Figure S9 (training data coverage) -- reads data/v4/randomly/train.nc:"
echo "        sbatch $SCRIPTS/run_figure_S9.sh"
echo "Finished: $(date)"
ls -la $OUTDIR/
