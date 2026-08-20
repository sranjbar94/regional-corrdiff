"""
figure_S5_ensemble_sensitivity.py — Figure S5: CRPS sensitivity to ensemble
size, computed from the ACTUAL 16 ensemble members in the raw chunk files
(no simulation/assumption -- every reported CRPS(N) value below is the real
fair-CRPS estimator evaluated on a genuine N-member subset drawn from the
16 members that were actually generated).

THIS IS A SEPARATE, SLOW SCRIPT. It re-reads every chunk file's full
(16, T, H, W) prediction array for T2m, u10, and Rain, which is far heavier
than the rest of the figure pipeline (~10 min). Run it on its own and cache
the result to a small JSON so repeated figure-style tweaks don't require
re-reading the chunks every time.

Method
------
For each target ensemble size N in {1,2,4,8,16}:
  - if N == 16: use all 16 members directly (one CRPS value, no subsampling
    variance to average over).
  - if N < 16: draw `n_subsets` independent random subsets of size N from
    the 16 available members (without replacement within a subset), compute
    fair-CRPS for each subset, and report the mean +/- std across subsets.
    This is the standard way to estimate "CRPS at ensemble size N" from a
    larger ensemble without retraining/regenerating members at smaller N.

CRPS estimator: identical fair-CRPS formula already used in score_g1.py /
score_g2_corrdiff.py (mean pairwise absolute difference correction), so
numbers here are directly comparable to the main results tables.

Usage:
    python3 figure_S5_ensemble_sensitivity.py \
        --g1-chunks outputs/figure_data_g1 \
        --g2-chunks outputs/figure_data_g2_corrdiff \
        --n-chunks 60 \
        --n-subsets 20 \
        --outdir paper_figures_v2 \
        --cache paper_figures_v2/figure_S5_cache.json

Re-plot only (skip the slow chunk read) once a cache exists:
    python3 figure_S5_ensemble_sensitivity.py --from-cache paper_figures_v2/figure_S5_cache.json \
        --outdir paper_figures_v2
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_style, CB, DC, save_fig, panel_label

ENSEMBLE_SIZES = [1, 2, 4, 8, 16]


def fair_crps(truth_flat, ens_flat):
    """
    truth_flat: (N,)
    ens_flat:   (E, N)
    Identical formula to score_g1.py's crps_ensemble / score_g2_corrdiff.py's
    crps_subsample (mean |ens-truth| minus 0.5x mean pairwise |ens_i-ens_j|).
    """
    E = ens_flat.shape[0]
    mae_term = np.mean(np.abs(ens_flat - truth_flat[None, :]), axis=0)
    if E == 1:
        return float(np.mean(mae_term))
    spread_term = np.zeros(truth_flat.shape[0], dtype=np.float64)
    for i in range(E):
        for j in range(i+1, E):
            spread_term += np.abs(ens_flat[i] - ens_flat[j])
    spread_term /= (E*(E-1)/2)
    return float(np.mean(mae_term - 0.5*spread_term))


def crps_vs_n(truth_flat, ens16_flat, n_subsets=20, seed=0):
    """
    truth_flat:  (N_pixels,)
    ens16_flat:  (16, N_pixels)  -- the real 16 members
    Returns dict {ensemble_size: list_of_crps_values}, one value per random
    subset draw (or a single value if ensemble_size == E_total, since there
    is only one possible "subset" in that case). Raw values are returned
    (not pre-averaged) so the caller can pool them correctly across chunks
    before computing mean/std -- averaging per-chunk first and then taking
    std-of-means would understate/misrepresent the true subsampling spread.
    """
    rng = np.random.default_rng(seed)
    E_total = ens16_flat.shape[0]
    out = {}
    for n in ENSEMBLE_SIZES:
        if n > E_total:
            continue
        if n == E_total:
            out[n] = [fair_crps(truth_flat, ens16_flat)]
            continue
        vals = []
        for _ in range(n_subsets):
            members = rng.choice(E_total, size=n, replace=False)
            vals.append(fair_crps(truth_flat, ens16_flat[members]))
        out[n] = vals
    return out


def collect_from_chunks(chunk_dir, var_key, n_chunks, n_subsets, pixel_stride,
                        seed=0):
    """
    Reads up to n_chunks chunk files for one variable, accumulates per-size
    CRPS across all chunks (each chunk contributes its own subsample draws,
    pooled together at the end as a simple unweighted mean across chunks --
    every chunk has the same number of (time x space) elements so this is
    equivalent to pooling all pixels first).
    """
    import netCDF4 as nc4
    files = sorted(Path(chunk_dir).glob("chunk_*.nc"),
                   key=lambda p: int(p.stem.split("_")[1]))[:n_chunks]
    if not files:
        raise FileNotFoundError(f"No chunks in {chunk_dir}")

    per_size_vals = {n: [] for n in ENSEMBLE_SIZES}
    t0 = time.time()
    for ci, cf in enumerate(files):
        ds = nc4.Dataset(cf, "r")
        try:
            truth = np.array(ds.groups["truth"].variables[var_key][:],
                             dtype=np.float32)              # (T,H,W)
            pred  = np.array(ds.groups["prediction"].variables[var_key][:],
                             dtype=np.float32)               # (E,T,H,W)
        finally:
            ds.close()

        E = pred.shape[0]
        if E < max(ENSEMBLE_SIZES):
            print(f"    WARNING: {cf.name} has only {E} members "
                  f"(<{max(ENSEMBLE_SIZES)}); sizes above {E} skipped for this chunk")

        truth_flat = truth[:, ::pixel_stride, ::pixel_stride].reshape(-1)
        pred_flat  = pred[:, :, ::pixel_stride, ::pixel_stride].reshape(E, -1)

        sizes_here = crps_vs_n(truth_flat, pred_flat, n_subsets=n_subsets,
                               seed=seed+ci)
        # Pool the RAW per-subset values across chunks (not a per-chunk mean)
        # so the final std reflects genuine member-subsampling variance, not
        # chunk-to-chunk (storm-to-storm) variance.
        for n, vals in sizes_here.items():
            per_size_vals[n].extend(vals)

        if (ci+1) % 10 == 0 or ci == len(files)-1:
            elapsed = time.time()-t0
            print(f"    {var_key}: chunk {ci+1}/{len(files)} "
                  f"({elapsed:.1f}s elapsed)")

    result = {}
    for n, vals in per_size_vals.items():
        if vals:
            result[n] = {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n_chunks": len(files),
                        "n_values": len(vals)}
    return result


def run_collection(g1_chunks, g2_chunks, n_chunks, n_subsets, pixel_stride,
                   cache_path):
    targets = [
        ("T2m",  g1_chunks, "2t"),
        ("u10",  g1_chunks, "10u"),
        ("Rain", g2_chunks, "tp"),
    ]
    cache = {}
    for label, chunk_dir, key in targets:
        print(f"  [S5] collecting {label} ({key}) from {chunk_dir} "
              f"({n_chunks} chunks, {n_subsets} subsets/size)...")
        cache[label] = collect_from_chunks(chunk_dir, key, n_chunks,
                                           n_subsets, pixel_stride)
        print(f"  [S5] {label} done: "
              + ", ".join(f"N={n}:{v['mean']:.4f}" for n,v in
                          sorted(cache[label].items())))
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  [S5] cache written -> {cache_path}")
    return cache


def make_figure(cache, outdir):
    apply_style()
    import matplotlib.pyplot as plt

    panels = [("T2m", r"T2m CRPS (K)"),
              ("u10", r"u10 CRPS (m s$^{-1}$)"),
              ("Rain", r"Rain CRPS (mm hr$^{-1}$)")]

    fig, axes = plt.subplots(1, 3, figsize=(DC*0.95, 2.8),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18,
                        wspace=0.35)

    for pi, (label, ylabel) in enumerate(panels):
        ax = axes[pi]
        if label not in cache:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        d = cache[label]
        ns = sorted(int(n) for n in d.keys())
        means = [d[str(n)]["mean"] if str(n) in d else d[n]["mean"] for n in ns]
        stds  = [d[str(n)]["std"]  if str(n) in d else d[n]["std"]  for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", ms=4, lw=1.3,
                   color=CB["corrdiff"], capsize=3, ecolor=CB["grey"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ENSEMBLE_SIZES)
        ax.set_xticklabels([str(n) for n in ENSEMBLE_SIZES])
        ax.set_xlabel("Ensemble size N", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(label, fontsize=9, pad=4)
        ax.tick_params(labelsize=7)
        panel_label(ax, chr(97+pi))

    fig.suptitle(
        "CRPS Sensitivity to Ensemble Size (real 16-member subsampling)",
        fontsize=10, fontweight="bold", y=0.99)
    save_fig(fig, f"{outdir}/figure_S5_ensemble_sensitivity.pdf")
    plt.close(fig)
    print("Figure S5 done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g1-chunks", default="outputs/figure_data_g1")
    p.add_argument("--g2-chunks", default="outputs/figure_data_g2_corrdiff")
    p.add_argument("--n-chunks", type=int, default=60,
                   help="Max number of chunk files to read per variable "
                        "(60 = full eval set; reduce for a quick test run)")
    p.add_argument("--n-subsets", type=int, default=20,
                   help="Random member-subsets averaged per (chunk, N<16)")
    p.add_argument("--pixel-stride", type=int, default=4,
                   help="Spatial subsampling stride to keep memory/runtime "
                        "bounded (4 -> uses 1/16 of pixels per field)")
    p.add_argument("--outdir", default="paper_figures_v2")
    p.add_argument("--cache", default=None,
                   help="Path to write/read the JSON cache of computed CRPS "
                        "values. Default: <outdir>/figure_S5_cache.json")
    p.add_argument("--from-cache", default=None,
                   help="Skip chunk reading entirely and re-plot from this "
                        "existing cache JSON (fast iteration on figure style).")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    cache_path = a.cache or f"{a.outdir}/figure_S5_cache.json"

    if a.from_cache:
        print(f"  [S5] loading cache from {a.from_cache} (skipping chunk read)")
        with open(a.from_cache) as f:
            cache = json.load(f)
    else:
        t0 = time.time()
        cache = run_collection(a.g1_chunks, a.g2_chunks, a.n_chunks,
                               a.n_subsets, a.pixel_stride, cache_path)
        print(f"  [S5] total collection time: {time.time()-t0:.1f}s")

    make_figure(cache, a.outdir)

if __name__ == "__main__":
    main()
