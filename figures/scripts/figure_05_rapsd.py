"""
figure_05_rapsd.py — Figure 5: RAPSD spectral analysis (corrected).

Matches reference style: thick "target" (HRRR) band, overlaid method lines,
clean power-law spectra with no Nyquist cliff (windowed + detrended FFT).

One panel per variable. Default: u10, T2m, Rain (3 panels, 1 row).
Optionally add more with --vars.

Run from project root:
    python paper_scripts_v2/figure_05_rapsd.py \
        --g1 figure_data_g1/eval_g1_main.nc \
        --g2 figure_data_g2/eval_g2_corrdiff_main.nc \
        --g3 figure_data_g3/eval_g3_main.nc \
        --outdir paper_figures_v2
"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, CB, DC, save_fig, panel_label, proxy_footnote)
from _rapsd_core import mean_rapsd


# (label, unit-for-spectra, nc_key, group_source, is_wind)
PANEL_DEFS = {
    "u10":  ("u10",  r"m$^2$ s$^{-2}\cdot$km", "10u",  "g1", True),
    "v10":  ("v10",  r"m$^2$ s$^{-2}\cdot$km", "10v",  "g1", False),
    "T2m":  ("T2m",  r"K$^2\cdot$km",          "2t",   "g1", False),
    "sp":   (r"$p_s$",r"Pa$^2\cdot$km",        "sp",   "g1", False),
    "q":    ("q",    r"(kg kg$^{-1}$)$^2\cdot$km","q", "g1", False),
    "Rain": ("Rain", r"(mm hr$^{-1}$)$^2\cdot$km","tp","g2", False),
    "sf":   ("Snow", r"(mm hr$^{-1}$)$^2\cdot$km","sf","g2", False),
    "ssrd": (r"SW$\downarrow$", r"(W m$^{-2}$)$^2\cdot$km","ssrd","g3", False),
    "strd": (r"LW$\downarrow$", r"(W m$^{-2}$)$^2\cdot$km","strd","g3", False),
}


def make_figure(nc_map, var_list, outdir, n_samples=200):
    apply_style()
    n = len(var_list)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DC, 2.7*nrows),
                             squeeze=False,
                             constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.16,
                        hspace=0.42, wspace=0.30)

    for pi, vkey in enumerate(var_list):
        ax = axes.ravel()[pi]
        label, unit, nc_key, gsrc, wind = PANEL_DEFS[vkey]
        nc_path = nc_map[gsrc]
        print(f"  RAPSD {label}...")

        try:
            # Target (HRRR) — thick band
            wl, ps_truth = mean_rapsd(nc_path, nc_key, "truth",
                                      n_samples=n_samples, wind=wind)
            # CorrDiff mean
            _,  ps_cd    = mean_rapsd(nc_path, nc_key, "pred_mean",
                                      n_samples=n_samples, wind=wind)
            # CorrDiff member (random)
            _,  ps_mem   = mean_rapsd(nc_path, nc_key, "pred_random",
                                      n_samples=min(n_samples,80), wind=wind)
            # UNet baseline (real regression-only output)
            _,  ps_unet  = mean_rapsd(nc_path, nc_key, "pred_unet",
                                      n_samples=n_samples, wind=wind)
            # ERA5 baseline (real bilinear-interpolated ERA5)
            _,  ps_era5  = mean_rapsd(nc_path, nc_key, "pred_era5",
                                      n_samples=min(n_samples,80), wind=wind)

            # Plot: target as thick band behind (40% narrower: lw 5 -> 3)
            ax.plot(wl, ps_truth, color="#F0E442", lw=3.3, alpha=0.9,
                    solid_capstyle="round", label="HRRR (target)", zorder=2)
            ax.plot(wl, ps_era5, color="black", lw=1.4, label="ERA5", zorder=3)
            ax.plot(wl, ps_unet, color=CB["unet"], lw=1.0, ls="--",
                    label="UNet", zorder=4)
            ax.plot(wl, ps_cd, color="#1565C0", lw=1.4, ls="-.",
                    label="CorrDiff ensemble", zorder=6)
            ax.plot(wl, ps_mem, color="#2196F3", lw=1.4, ls=":",
                    alpha=0.9, label="CorrDiff member", zorder=5)

            ax.set_xscale("log"); ax.set_yscale("log")
            # Reference style: wavenumber on x (1/km), so invert to wavenumber
            # Keep wavelength for interpretability but reverse axis so small
            # scales are on the right (matches energy cascade convention).
            ax.invert_xaxis()

            # ERA5 resolution marker (~25 km)
            ax.axvline(25, color=CB["grey"], lw=0.7, ls=":", zorder=1)

        except Exception as e:
            print(f"  RAPSD {label} failed: {e}")
            ax.text(0.5, 0.5, "data error", transform=ax.transAxes,
                    ha="center", va="center")

        ax.set_xlabel("Wavelength (km)", fontsize=8)
        ax.set_ylabel(f"PSD ({unit})", fontsize=7.5)
        ax.set_title(label, fontsize=9, pad=4)
        panel_label(ax, chr(97+pi))
        if pi == 0:
            ax.legend(fontsize=6, loc="lower left", framealpha=0.9)

    # hide unused axes
    for j in range(n, nrows*ncols):
        axes.ravel()[j].axis("off")

    proxy_footnote(fig)
    # Rename existing file to _old before saving the new version
    import shutil as _sh
    old_path = f"{outdir}/figure_05_rapsd.pdf"
    old_png  = f"{outdir}/figure_05_rapsd.png"
    if Path(old_path).exists():
        _sh.copy2(old_path, f"{outdir}/figure_05_rapsd_old.pdf")
    if Path(old_png).exists():
        _sh.copy2(old_png, f"{outdir}/figure_05_rapsd_old.png")
    save_fig(fig, f"{outdir}/figure_05_rapsd.pdf")
    plt.close(fig)
    print("Figure 5 done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g1", default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--g2", default="figure_data_g2/eval_g2_corrdiff_main.nc")
    p.add_argument("--g3", default="figure_data_g3/eval_g3_main.nc")
    p.add_argument("--vars", nargs="*",
                   default=["u10","v10","T2m","sp","q","Rain","sf","ssrd","strd"],
                   help="Variables to plot (keys in PANEL_DEFS)")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--outdir", default="paper_figures_v2")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    nc_map = {"g1": a.g1, "g2": a.g2, "g3": a.g3}
    make_figure(nc_map, a.vars, a.outdir, a.n_samples)

if __name__ == "__main__":
    main()
