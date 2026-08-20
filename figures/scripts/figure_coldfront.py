"""
figure_coldfront.py — Cold-front cross-section study (reference-style).

Replicates the attached multi-panel front analysis:
  Rows (3): 2-m Temperature | Along-front wind | Across-front wind
  Cols (4): ERA5 | CorrDiff mean | HRRR (target) | NW-SE cross section
  - Spatial panels show the field + wind-vector arrows (quiver)
  - A thin dashed transect line is drawn across each contour panel
  - Cross-section column averages 21 parallel lines along the transect;
    shows ERA5 (red), HRRR (black), CorrDiff mean (orange) ± 1 std shading

The "front" sample should be a land-ocean case with a sharp gradient and
high ensemble spread. Pass --idx after picking with the storm-finder.

Run from project root:
    python scripts/figure_coldfront.py \
        --g1 ../outputs/figure_data_g1/eval_g1_main.nc \
        --idx 2445 --outdir output
"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, map_coordinates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, CB, DC, save_fig, load_nc_var,
                       proxy_footnote)


def get_components(nc_path, group, idx):
    """Return (t2m, u, v) for a sample."""
    t = load_nc_var(nc_path, group, "2t", idx)
    u = load_nc_var(nc_path, group, "10u", idx)
    v = load_nc_var(nc_path, group, "10v", idx)
    return np.array(t), np.array(u), np.array(v)


def rotate_wind(u, v, angle_deg):
    """Rotate wind into front-relative coords. Returns (along, across)."""
    a = np.deg2rad(angle_deg)
    along  =  u*np.cos(a) + v*np.sin(a)
    across = -u*np.sin(a) + v*np.cos(a)
    return along, across


def sample_transect(field, p0, p1, n=100):
    """Bilinear-sample field along a line from p0 to p1. p=(row,col)."""
    rr = np.linspace(p0[0], p1[0], n)
    cc = np.linspace(p0[1], p1[1], n)
    return map_coordinates(field, np.vstack([rr, cc]), order=1, mode="nearest")


def parallel_transects(field, p0, p1, n_lines=21, spacing=3, n=100):
    """Average field over n_lines parallel transects offset perpendicular."""
    dr = p1[0]-p0[0]; dc = p1[1]-p0[1]
    L = np.hypot(dr, dc)
    # perpendicular unit vector
    pr, pc = -dc/L, dr/L
    profiles = []
    for k in range(-(n_lines//2), n_lines//2+1):
        off_r = k*spacing*pr
        off_c = k*spacing*pc
        prof = sample_transect(field,
                               (p0[0]+off_r, p0[1]+off_c),
                               (p1[0]+off_r, p1[1]+off_c), n)
        profiles.append(prof)
    return np.array(profiles)   # (n_lines, n)


def make_figure(nc_path, idx, out_path, front_angle=45):
    apply_style()

    # ── Load fields ────────────────────────────────────────────────────────
    t_truth, u_truth, v_truth = get_components(nc_path, "truth", idx)
    t_cd,    u_cd,    v_cd     = get_components(nc_path, "pred_mean", idx)
    # ERA5 baseline (real bilinear-interpolated ERA5)
    t_era5, u_era5, v_era5 = get_components(nc_path, "pred_era5", idx)
    # UNet baseline (real regression-only output)
    t_unet, u_unet, v_unet = get_components(nc_path, "pred_unet", idx)

    # Ensemble members for std in cross-section (use available 4)
    members = ["pred_mean","pred_median","pred_random","pred_closest"]
    ens_t = np.stack([np.array(load_nc_var(nc_path,m,"2t",idx)) for m in members])
    ens_u = np.stack([np.array(load_nc_var(nc_path,m,"10u",idx)) for m in members])
    ens_v = np.stack([np.array(load_nc_var(nc_path,m,"10v",idx)) for m in members])

    # CorrDiff single random member (pred_random)
    t_member, u_member, v_member = get_components(nc_path, "pred_random", idx)
    al_member, ac_member = rotate_wind(u_member, v_member, front_angle)

    # Front-relative wind components
    al_truth, ac_truth = rotate_wind(u_truth, v_truth, front_angle)
    al_cd,    ac_cd     = rotate_wind(u_cd,    v_cd,    front_angle)
    al_era5,  ac_era5   = rotate_wind(u_era5,  v_era5,  front_angle)
    al_unet,  ac_unet   = rotate_wind(u_unet,  v_unet,  front_angle)

    # Transect line (NW → SE diagonal through center)
    H, W = t_truth.shape
    p0 = (int(0.20*H), int(0.20*W))   # NW
    p1 = (int(0.80*H), int(0.80*W))   # SE

    # ── Row definitions ──────────────────────────────────────────────────────
    # spatial order: ERA5 | UNet | CorrDiff mean | CorrDiff member | HRRR
    rows = [
        dict(title="2-m Temperature", unit="K", cmap="RdYlBu_r",
             f_t=t_truth, f_c=t_cd, f_m=t_member, f_e=t_era5, f_u=t_unet,
             ens=ens_t,
             au_t=u_truth, av_t=v_truth, au_c=u_cd, av_c=v_cd,
             au_m=u_member, av_m=v_member,
             au_e=u_era5, av_e=v_era5, au_u=u_unet, av_u=v_unet),
        dict(title="V-Wind", unit=r"m s$^{-1}$", cmap="RdBu_r",
             f_t=al_truth, f_c=al_cd, f_m=al_member, f_e=al_era5, f_u=al_unet,
             ens=np.stack([rotate_wind(ens_u[k],ens_v[k],front_angle)[0]
                           for k in range(len(members))]),
             au_t=al_truth, av_t=np.zeros_like(al_truth),
             au_c=al_cd, av_c=np.zeros_like(al_cd),
             au_m=al_member, av_m=np.zeros_like(al_member),
             au_e=al_era5, av_e=np.zeros_like(al_era5),
             au_u=al_unet, av_u=np.zeros_like(al_unet)),
        dict(title="U-Wind", unit=r"m s$^{-1}$", cmap="RdBu_r",
             f_t=ac_truth, f_c=ac_cd, f_m=ac_member, f_e=ac_era5, f_u=ac_unet,
             ens=np.stack([rotate_wind(ens_u[k],ens_v[k],front_angle)[1]
                           for k in range(len(members))]),
             au_t=np.zeros_like(ac_truth), av_t=ac_truth,
             au_c=np.zeros_like(ac_cd), av_c=ac_cd,
             au_m=np.zeros_like(ac_member), av_m=ac_member,
             au_e=np.zeros_like(ac_era5), av_e=ac_era5,
             au_u=np.zeros_like(ac_unet), av_u=ac_unet),
    ]

    # 5 spatial cols: ERA5 | UNet | CorrDiff mean | CorrDiff member | HRRR
    COL_LABELS = ["ERA5", "UNet", "CorrDiff mean", "CorrDiff member", "HRRR (target)"]

    fig = plt.figure(figsize=(DC*1.75, 6.6), constrained_layout=False)
    gs = gridspec.GridSpec(3, 8, figure=fig,
                           width_ratios=[1,1,1,1,1,0.06,0.55,1.25],
                           hspace=0.26, wspace=0.10,
                           left=0.045, right=0.97, top=0.88, bottom=0.07)

    skip = 16   # quiver subsample
    yy, xx = np.mgrid[0:H:skip, 0:W:skip]

    for ri, R in enumerate(rows):
        # shared color scale across the 5 spatial panels (from truth)
        if R["cmap"] == "RdBu_r":
            vlim = float(np.nanpercentile(np.abs(R["f_t"]), 98))
            norm = mcolors.Normalize(-vlim, vlim)
        else:
            p2,p98 = np.nanpercentile(R["f_t"],[2,98])
            norm = mcolors.Normalize(p2, p98)

        # ERA5 | UNet | CorrDiff mean | CorrDiff member | HRRR
        spatial = [
            (R["f_e"], R["au_e"], R["av_e"]),   # 0 ERA5
            (R["f_u"], R["au_u"], R["av_u"]),   # 1 UNet
            (R["f_c"], R["au_c"], R["av_c"]),   # 2 CorrDiff mean
            (R["f_m"], R["au_m"], R["av_m"]),   # 3 CorrDiff member
            (R["f_t"], R["au_t"], R["av_t"]),   # 4 HRRR truth
        ]
        N_SPATIAL = len(spatial)
        im = None
        for ci, (field, au, av) in enumerate(spatial):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(field, origin="lower", cmap=R["cmap"], norm=norm,
                           interpolation="nearest", aspect="equal")
            ax.quiver(xx, yy, au[::skip,::skip], av[::skip,::skip],
                      scale=300, width=0.004, color="#222", alpha=0.7)
            ax.plot([p0[1],p1[1]],[p0[0],p1[0]], "k--", lw=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            # border colour per column
            if ci == 4:
                border = CB["truth"]
            elif ci == 3:
                border = CB["corrdiff"]   # member same colour as mean
            elif ci == 2:
                border = CB["corrdiff"]
            elif ci == 1:
                border = CB["unet"]
            else:
                border = CB["era5"]
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border)
                sp.set_linewidth(1.4 if ci == N_SPATIAL-1 else 0.7)
            if ri == 0:
                ax.set_title(COL_LABELS[ci], fontsize=10, color=border,
                             fontweight="bold" if ci == N_SPATIAL-1 else "normal",
                             pad=6)
            if ci == 0:
                ax.set_ylabel(R["title"], fontsize=10, labelpad=4)

        # colorbar — now at column index 5
        cax = fig.add_subplot(gs[ri, 5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(R["unit"], fontsize=8.5, labelpad=2)
        cb.ax.tick_params(labelsize=6, pad=1)

        # spacer at index 6 (left empty)
        # cross-section panel at index 7
        axc = fig.add_subplot(gs[ri, 7])
        n_pts = 100
        dist = np.linspace(0, np.hypot(p1[0]-p0[0], p1[1]-p0[1])*3.0/1000*1000,
                           n_pts)  # km along transect (3km/px)
        dist = np.linspace(0, np.hypot(p1[0]-p0[0], p1[1]-p0[1])*3.0, n_pts)

        prof_truth  = parallel_transects(R["f_t"], p0, p1).mean(0)
        prof_era5   = parallel_transects(R["f_e"], p0, p1).mean(0)
        prof_unet   = parallel_transects(R["f_u"], p0, p1).mean(0)
        prof_member = parallel_transects(R["f_m"], p0, p1).mean(0)
        # CorrDiff ensemble mean + std across members along transect
        ens_profiles = np.stack([parallel_transects(R["ens"][k], p0, p1).mean(0)
                                 for k in range(R["ens"].shape[0])])
        prof_cd_mean = ens_profiles.mean(0)
        prof_cd_std  = ens_profiles.std(0)

        axc.plot(dist, prof_truth,  color="black",       lw=1.3, label="HRRR")
        axc.plot(dist, prof_era5,   color=CB["era5"],    lw=1.1, label="ERA5")
        axc.plot(dist, prof_unet,   color=CB["unet"],    lw=1.1, ls="--", label="UNet")
        axc.plot(dist, prof_cd_mean,color=CB["corrdiff"],lw=1.3, label="CorrDiff mean")
        axc.fill_between(dist, prof_cd_mean-prof_cd_std, prof_cd_mean+prof_cd_std,
                         color=CB["corrdiff"], alpha=0.25, lw=0)
        axc.plot(dist, prof_member, color=CB["corrdiff"],lw=0.9, ls=":",
                 label="CorrDiff member")
        axc.set_ylabel(R["unit"], fontsize=9, labelpad=2)
        if ri==2:
            axc.set_xlabel("Distance along NW–SE transect (km)", fontsize=9)
        if ri==0:
            axc.legend(fontsize=8, loc="best")
            axc.set_title("NW–SE cross section", fontsize=10, pad=4)
        axc.tick_params(labelsize=8)

    fig.suptitle(
        f"Cold-front downscaling case study (sample {idx})",
        fontsize=12, fontweight="bold", y=0.995)
    proxy_footnote(fig)
    save_fig(fig, out_path)
    plt.close(fig)
    print("Cold-front figure done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g1", default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--idx", type=int, default=2445,
                   help="Single sample index (produces figure_coldfront_case.pdf)")
    p.add_argument("--idxs", type=int, nargs="*", default=None,
                   help="Multiple indices -> figure_coldfront_a/b/c. Overrides --idx.")
    p.add_argument("--angle", type=float, default=45,
                   help="Front orientation angle (deg). Single value or one per idx.")
    p.add_argument("--angles", type=float, nargs="*", default=None,
                   help="Per-case angles matching --idxs.")
    p.add_argument("--outdir", default="paper_figures_v2")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    if a.idxs:
        angles = a.angles if a.angles else [a.angle]*len(a.idxs)
        for letter, idx, ang in zip("abc", a.idxs, angles):
            print(f"  Cold-front {letter}: idx {idx}, angle {ang}...")
            make_figure(a.g1, idx,
                        f"{a.outdir}/figure_coldfront_{letter}.pdf", ang)
    else:
        make_figure(a.g1, a.idx, f"{a.outdir}/figure_coldfront_case.pdf", a.angle)

if __name__ == "__main__":
    main()
