"""
figures_supp.py — Supplementary Figures S1-S5.

S1: Scatter (prediction vs HRRR), 9 vars, R²+RMSE
S2: Q-Q plots (u10, T2m, sp, tp)
S3: Regional skill maps (placeholder — needs coords)
S4: Training data coverage (placeholder — needs HPC training NC)
S5: Ensemble size sensitivity (from 4 pseudo-members)

Run from project root:
    python paper_scripts_v2/figures_supp.py \
        --g1 figure_data_g1/eval_g1_main.nc \
        --g2 figure_data_g2/eval_g2_corrdiff_main.nc \
        --g3 figure_data_g3/eval_g3_main.nc \
        --outdir paper_figures_v2
"""
import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, CB, CMAP, DC, save_fig, panel_label,
                       proxy_footnote, VARS_G1, VARS_G2, VARS_G3, ALL_VARS,
                       NC_GROUP, load_nc_var, wind_speed_nc)


# ═══ S1 — Scatter (a: sample-mean, b: all-pixel) ═══════════════════════════
def _s1_pairs_mean(ncp,key,n=3000,seed=42):
    """One (pred_mean, truth) value per sample (spatial mean)."""
    import netCDF4 as nc4
    rng=np.random.default_rng(seed); ds=nc4.Dataset(ncp,"r")
    N=ds.groups["truth"].variables[key].shape[0]
    idx=rng.choice(N,min(n,N),replace=False)
    pm=np.array(ds.groups["pred_mean"].variables[key][idx]).reshape(len(idx),-1).mean(1)
    tr=np.array(ds.groups["truth"].variables[key][idx]).reshape(len(idx),-1).mean(1)
    ds.close(); return pm,tr


def _s1_pairs_pixel(ncp,key,n_samples=400,n_pix=400000,seed=42):
    """All-pixel (pred_mean, truth) pairs, subsampled across n_samples patches."""
    import netCDF4 as nc4
    rng=np.random.default_rng(seed); ds=nc4.Dataset(ncp,"r")
    N=ds.groups["truth"].variables[key].shape[0]
    idx=rng.choice(N,min(n_samples,N),replace=False)
    pm=np.array(ds.groups["pred_mean"].variables[key][idx]).ravel()
    tr=np.array(ds.groups["truth"].variables[key][idx]).ravel()
    ds.close()
    # subsample pixels for plotting tractability
    if len(pm)>n_pix:
        sel=rng.choice(len(pm),n_pix,replace=False)
        pm,tr=pm[sel],tr[sel]
    return pm,tr


def _s1_render(pairs_fn, title_suffix, out_path, g1,g2,g3):
    apply_style()
    nc={"g1":g1,"g2":g2,"g3":g3}
    gc={"g1":CB["g1"],"g2":CB["g2"],"g3":CB["g3"]}
    fig,axes=plt.subplots(3,3,figsize=(DC,DC*0.95),
                           gridspec_kw={"hspace":0.45,"wspace":0.40})
    for i,vd in enumerate(ALL_VARS):
        ax=axes.ravel()[i]; grp=NC_GROUP[vd["key"]]; ec=gc[grp]
        try: pm,tr=pairs_fn(nc[grp],vd["key"])
        except Exception as e:
            print(f"  S1 {vd['key']}: {e}")
            ax.text(0.5,0.5,"N/A",transform=ax.transAxes,ha="center"); continue
        log=(vd["key"] in ("tp","sf"))
        if log: pm=np.log10(np.clip(pm,1e-3,None)); tr=np.log10(np.clip(tr,1e-3,None))
        ax.hexbin(tr,pm,gridsize=40,cmap=CMAP["density"],mincnt=1,
                  linewidths=0.1,bins="log")
        lims=[min(pm.min(),tr.min()),max(pm.max(),tr.max())]
        ax.plot(lims,lims,"k--",lw=0.8,zorder=5)
        r2=1-np.sum((tr-pm)**2)/np.sum((tr-tr.mean())**2)
        rm=float(np.sqrt(np.mean((pm-tr)**2)))
        ax.text(0.05,0.95,f"$R^2$={r2:.3f}\nRMSE={rm:.3g}",transform=ax.transAxes,
                ha="left",va="top",fontsize=6,color=ec,
                bbox=dict(boxstyle="round,pad=0.15",fc="white",alpha=0.7,lw=0))
        pre="log " if log else ""
        ax.set_xlabel(f"{pre}HRRR ({vd['unit']})",fontsize=6.5)
        ax.set_ylabel(f"{pre}CorrDiff ({vd['unit']})",fontsize=6.5)
        ax.set_title(vd["label"],fontsize=8,pad=3,color=ec,fontweight="bold")
        panel_label(ax,chr(97+i),x=-0.18,y=1.05)
    fig.suptitle(title_suffix,fontsize=9,y=1.0)
    save_fig(fig,out_path); plt.close(fig)


def s1_scatter(g1,g2,g3,outdir):
    # S1a — sample-mean (one point per test sample)
    _s1_render(_s1_pairs_mean,
               "CorrDiff mean vs HRRR — sample spatial means (n=3000)",
               f"{outdir}/figure_S1a_scatter.pdf", g1,g2,g3)
    print("S1a (sample-mean) done.")
    # S1b — all-pixel (256x256 per sample, subsampled)
    _s1_render(_s1_pairs_pixel,
               "CorrDiff mean vs HRRR — all pixels (subsampled)",
               f"{outdir}/figure_S1b_scatter.pdf", g1,g2,g3)
    print("S1b (all-pixel) done.")




# ═══ S2 — Q-Q ══════════════════════════════════════════════════════════════
def s2_qq(g1,g2,outdir):
    apply_style()
    def fs(nc,grp,var,n=200000,seed=42):
        rng=np.random.default_rng(seed)
        a=load_nc_var(nc,grp,var).ravel(); return rng.choice(a,min(n,len(a)),replace=False)
    def fw(nc,grp,n=200000,seed=42):
        rng=np.random.default_rng(seed); w=wind_speed_nc(nc,grp).ravel()
        return rng.choice(w,min(n,len(w)),replace=False)
    panels=[("(a) u10",r"m s$^{-1}$",False,g1,"wind"),
            ("(b) T2m","K",False,g1,"2t"),
            ("(c) $p_s$","Pa",False,g1,"sp"),
            ("(d) Rain",r"mm hr$^{-1}$",True,g2,"tp")]
    fig,axes=plt.subplots(1,4,figsize=(DC*1.1,2.8),
                           gridspec_kw={"wspace":0.42},constrained_layout=False)
    fig.subplots_adjust(left=0.07,right=0.98,top=0.87,bottom=0.18)
    ql=np.linspace(0,100,500)
    for pi,(title,unit,log,nc,key) in enumerate(panels):
        ax=axes[pi]
        try:
            if key=="wind": td=fw(nc,"truth"); md=fw(nc,"pred_mean")
            else: td=fs(nc,"truth",key); md=fs(nc,"pred_mean",key)
            tq=np.percentile(td,ql); mq=np.percentile(md,ql)
            if log:
                m=(tq>0)&(mq>0); ax.loglog(tq[m],mq[m],color=CB["corrdiff"],lw=1.3)
                ax.loglog([tq[m].min(),tq[m].max()],[tq[m].min(),tq[m].max()],"k:",lw=0.8)
            else:
                ax.plot(tq,mq,color=CB["corrdiff"],lw=1.3,label="CorrDiff")
                ax.plot([tq[0],tq[-1]],[tq[0],tq[-1]],"k:",lw=0.8,label="y=x")
            for pct in ([90,95,99] if log else [75,90,99]):
                j=np.argmin(np.abs(ql-pct)); ax.axvline(tq[j],color=CB["grey_lt"],lw=0.5,ls=":")
        except Exception as e: print(f"  s2 {key}: {e}")
        ax.set_xlabel(f"HRRR ({unit})",fontsize=7)
        if pi==0: ax.set_ylabel(f"CorrDiff ({unit})",fontsize=7); ax.legend(fontsize=6)
        ax.set_title(title,fontsize=8,pad=3); panel_label(ax,chr(97+pi))
    save_fig(fig,f"{outdir}/figure_S2_qq.pdf"); plt.close(fig)
    print("S2 done.")


# ═══ S3 — Regional maps (placeholder) ══════════════════════════════════════
def _load_latlon(latlon_json):
    """Load the eval_idx -> lat/lon/basin lookup built by build_latlon_lookup_g*.py.
    Returns arrays aligned to eval_idx order, with NaN for any unmatched sample."""
    with open(latlon_json) as f:
        records = json.load(f)
    n = max(r["eval_idx"] for r in records) + 1
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    basin = np.array(["unmatched"] * n, dtype=object)
    for r in records:
        if r.get("matched"):
            i = r["eval_idx"]
            lat[i] = r["lat"]
            lon[i] = r["lon"]
            basin[i] = r["basin"]
    return lat, lon, basin


def _per_sample_error(nc_path, key, n=None, seed=7, wind=False):
    """Per-sample RMSE (pred_mean vs truth), spatially averaged within each
    sample, for every sample in the file (or a random subsample of n)."""
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path, "r")
    N = ds.groups["truth"].variables[key if not wind else "10u"].shape[0]
    idx = np.arange(N) if n is None else np.random.default_rng(seed).choice(N, min(n, N), replace=False)
    idx = np.sort(idx)
    if wind:
        tu = np.array(ds.groups["truth"].variables["10u"][idx])
        tv = np.array(ds.groups["truth"].variables["10v"][idx])
        pu = np.array(ds.groups["pred_mean"].variables["10u"][idx])
        pv = np.array(ds.groups["pred_mean"].variables["10v"][idx])
        truth = np.sqrt(tu**2 + tv**2)
        pred = np.sqrt(pu**2 + pv**2)
    else:
        truth = np.array(ds.groups["truth"].variables[key][idx])
        pred = np.array(ds.groups["pred_mean"].variables[key][idx])
    ds.close()
    rmse_per_sample = np.sqrt(((pred - truth) ** 2).mean(axis=(1, 2)))
    return idx, rmse_per_sample


def s3_regional(g1, g2, g1_latlon, g2_latlon, outdir):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DC, 6.4),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.28})

    panels = [
        ("T2m", g1, "2t", False, g1_latlon, CMAP["temp"], "K"),
        ("Rain (tp)", g2, "tp", False, g2_latlon, CMAP["precip"], "mm/hr"),
    ]

    for col, (label, ncp, key, wind, latlon_path, cmap, unit) in enumerate(panels):
        print(f"  S3 {label}...")
        try:
            lat_all, lon_all, basin_all = _load_latlon(latlon_path)
            idx, err = _per_sample_error(ncp, key, wind=wind)
            lat = lat_all[idx]
            lon = lon_all[idx]
            basin = basin_all[idx]
            valid = ~np.isnan(lat)
            lat, lon, err, basin = lat[valid], lon[valid], err[valid], basin[valid]

            # ── Top row: continuous scatter map ─────────────────────────────
            axm = axes[0, col]
            sc = axm.scatter(lon, lat, c=err, s=6, cmap=cmap, alpha=0.75,
                              linewidths=0, vmin=np.percentile(err, 2),
                              vmax=np.percentile(err, 98))
            cb = fig.colorbar(sc, ax=axm, fraction=0.05, pad=0.03)
            cb.set_label(f"RMSE ({unit})", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
            axm.set_xlabel("Longitude", fontsize=7)
            axm.set_ylabel("Latitude", fontsize=7)
            axm.set_title(f"({chr(97+col)}) {label} — sample RMSE map", fontsize=8, pad=4)
            axm.tick_params(labelsize=6)
            panel_label(axm, chr(97 + col))

            # ── Bottom row: basin-level summary bars ─────────────────────────
            axb = axes[1, col]
            basins_order = ["North Pacific Coast", "South Pacific Coast",
                             "North Atlantic Coast", "South Atlantic / Gulf Coast"]
            means, stds, ns = [], [], []
            for b in basins_order:
                m = basin == b
                if m.sum() > 0:
                    means.append(err[m].mean())
                    stds.append(err[m].std())
                    ns.append(int(m.sum()))
                else:
                    means.append(0); stds.append(0); ns.append(0)
            x = np.arange(len(basins_order))
            axb.bar(x, means, yerr=stds, capsize=3, color=CB["g1" if col == 0 else "g2"],
                    edgecolor="white", linewidth=0.4)
            for xi, n_b in zip(x, ns):
                axb.text(xi, 0.02, f"n={n_b}", transform=axb.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=5.5, color="#555")
            axb.set_xticks(x)
            axb.set_xticklabels([b.replace(" Coast", "").replace(" / ", "/\n")
                                  for b in basins_order], fontsize=6)
            axb.set_ylabel(f"RMSE ({unit})", fontsize=7)
            axb.set_title(f"({chr(99+col)}) {label} — RMSE by region", fontsize=8, pad=4)
            panel_label(axb, chr(99 + col))

        except Exception as e:
            print(f"  S3 {label} failed: {e}")
            for ax in (axes[0, col], axes[1, col]):
                ax.text(0.5, 0.5, "data error", transform=ax.transAxes,
                        ha="center", va="center")

    fig.suptitle("Fig S3 — Regional Skill Maps", fontsize=9, y=1.0)
    save_fig(fig, f"{outdir}/figure_S3_regional.pdf")
    plt.close(fig)
    print("S3 done.")




def main():
    p=argparse.ArgumentParser()
    p.add_argument("--g1",default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--g2",default="figure_data_g2/eval_g2_corrdiff_main.nc")
    p.add_argument("--g3",default="figure_data_g3/eval_g3_main.nc")
    p.add_argument("--g1-latlon",default="figure_data_g1/eval_g1_latlon.json")
    p.add_argument("--g2-latlon",default="figure_data_g2_corrdiff/eval_g2_corrdiff_latlon.json")
    p.add_argument("--outdir",default="paper_figures_v2")
    p.add_argument("--skip",nargs="*",default=[])
    a=p.parse_args()
    Path(a.outdir).mkdir(parents=True,exist_ok=True); sk=set(a.skip)
    if "s1" not in sk: s1_scatter(a.g1,a.g2,a.g3,a.outdir)
    if "s2" not in sk: s2_qq(a.g1,a.g2,a.outdir)
    if "s3" not in sk: s3_regional(a.g1,a.g2,a.g1_latlon,a.g2_latlon,a.outdir)
    # NOTE: S4 (training coverage) and S5 (ensemble-size CRPS sensitivity)
    # placeholders REMOVED. Real-data replacements are dedicated scripts:
    #   figure_S9_training_data.py             -> figure_S9_training_data.pdf
    #   figure_S5_ensemble_sensitivity.py       -> figure_S5_ensemble_sensitivity.pdf
    #     (run separately via run_figure_S5.sh -- reads raw 16-member chunks)
    print(f"\nSupplementary figures saved to {a.outdir}/")

if __name__=="__main__":
    main()
