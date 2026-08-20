"""
figure_01_spatial.py — Figure 1: Spatial comparison per variable.

One PDF per variable (7). Layout: 6 columns x 3 rows.
  Columns: ERA5 bilinear | UNet | CorrDiff Mean | CorrDiff Member | CorrDiff Std | HRRR
  Rows:    Storm | Moderate | Calm
Shared colorbar for the 5 field columns; separate colorbar for Std.

Run from project root:
    python paper_scripts_v2/figure_01_spatial.py \
        --g1 figure_data_g1/eval_g1_main.nc \
        --g2 figure_data_g2/eval_g2_corrdiff_main.nc \
        --g3 figure_data_g3/eval_g3_main.nc \
        --storm 2445 --moderate 1842 --calm 2236 \
        --outdir paper_figures_v2
"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, CB, CMAP, DC, FS_ANNOT, FS_TICK, save_fig,
                       load_nc_var, wind_speed_nc, load_baseline, proxy_footnote,
                       add_scalebar_note, neat_ticks)



def pick_variable_cases(nc_path, var_key, quantiles=(20,50,90), wind=False, seed=3):
    """
    Select one sample per quantile of this variable's spatial-mean distribution.
    Returns list of (label, idx). Low/mid/high quantiles → calm/moderate/storm.
    """
    import netCDF4 as nc4
    ds=nc4.Dataset(nc_path,"r")
    if wind:
        u=np.array(ds.groups["truth"]["10u"][:]); v=np.array(ds.groups["truth"]["10v"][:])
        field=np.sqrt(u**2+v**2)
    else:
        field=np.array(ds.groups["truth"][var_key][:])
    ds.close()
    means=field.reshape(field.shape[0],-1).mean(axis=1)
    labels={20:"Low (Q20)",50:"Median (Q50)",90:"High (Q90)",
            65:"Q65",80:"Q80",95:"High (Q95)"}
    out=[]
    for q in quantiles:
        target=np.percentile(means,q)
        idx=int(np.argmin(np.abs(means-target)))
        out.append((labels.get(q,f"Q{q}"), idx))
    return out


def rmse(p,t): return float(np.sqrt(np.nanmean((p-t)**2)))


def stamp(ax, txt, fs=FS_ANNOT-0.5):
    ax.text(0.96,0.05,txt,transform=ax.transAxes,ha="right",va="bottom",
            fontsize=fs,color="white",
            bbox=dict(boxstyle="round,pad=0.12",fc="black",alpha=0.55,lw=0))


def make_figure(var, nc_path, out_path):
    apply_style()
    key=var["key"]; unit=var["unit"]; cmap_field=var["cmap"]; log=var["log"]
    wind=(key in ("10u","10v"))   # treat wind comps with diverging map already

    wind_var=(key in ("10u","10v"))
    # Zero-dominated fields (radiation, precip) need higher quantiles so the
    # low/mid rows are not empty zero fields.
    if key in ("ssrd","strd","tp","sf"):
        q_sel=(95,80,65)
    else:
        q_sel=(90,50,20)
    cases=pick_variable_cases(nc_path, key, quantiles=q_sel, wind=wind_var)
    # cases ordered high→low so rows read High, Mid, Low

    def load_truth(i):
        try: return load_nc_var(nc_path,"truth",key,i)
        except: return np.zeros((256,256))
    def load_cd(grp,i):
        try: return load_nc_var(nc_path,grp,key,i)
        except: return np.zeros((256,256))

    COLS=[
        ("ERA5 bilinear",  lambda i:load_baseline(nc_path,"era5",key,i),CB["era5"],    False),
        ("UNet",           lambda i:load_baseline(nc_path,"unet",key,i),CB["unet"],    False),
        ("CorrDiff Mean", lambda i:load_cd("pred_mean",i),             CB["corrdiff"],False),
        ("CorrDiff Member",lambda i:load_cd("pred_random",i),          CB["member"],  False),
        ("HRRR Truth",    load_truth,                                   CB["truth"],   False),
        ("CorrDiff Std",  lambda i:load_cd("pred_std",i),              CB["std"],     True),
    ]

    truth_fields=[load_truth(ci) for _,ci in cases]
    allt=np.concatenate([f.ravel() for f in truth_fields])

    if log:
        vmin,vmax=0, max(float(np.nanpercentile(allt,99)),1.0)
        norm_field=mcolors.SymLogNorm(linthresh=0.1,linscale=0.5,vmin=0,vmax=vmax)
    elif cmap_field=="RdBu_r":  # diverging wind components, center 0
        vlim=float(np.nanpercentile(np.abs(allt),98))
        vmin,vmax=-vlim,vlim
        norm_field=mcolors.Normalize(vmin=vmin,vmax=vmax)
    else:
        p2,p98=np.nanpercentile(allt,2),np.nanpercentile(allt,98)
        c=np.nanmedian(allt); h=max(abs(p98-c),abs(c-p2))
        vmin,vmax=c-h,c+h
        norm_field=mcolors.Normalize(vmin=vmin,vmax=vmax)

    std_fields=[load_cd("pred_std",ci) for _,ci in cases]
    std_max=float(np.nanpercentile(np.concatenate([f.ravel() for f in std_fields]),95))
    norm_std=mcolors.Normalize(0,max(std_max,1e-6))

    N_COL=len(COLS); N_ROW=3
    fig=plt.figure(figsize=(DC*1.15, 4.5),constrained_layout=False)
    # Extra bottom margin for two horizontal colorbars stacked below panels
    gs=gridspec.GridSpec(N_ROW,N_COL,figure=fig,
                         hspace=0.10,wspace=0.06,
                         left=0.075,right=0.97,top=0.90,bottom=0.22)

    field_ims=[]
    for ri,(case_lbl,ci) in enumerate(cases):
        tf=truth_fields[ri]
        for mi,(col_lbl,loader,border,is_std) in enumerate(COLS):
            ax=fig.add_subplot(gs[ri,mi])
            field=loader(ci)
            cmap=CMAP["std"] if is_std else cmap_field
            norm=norm_std    if is_std else norm_field
            im=ax.imshow(field,origin="lower",cmap=cmap,norm=norm,
                         interpolation="nearest",aspect="equal")
            if not is_std: field_ims.append(im)
            ax.set_xticks([]); ax.set_yticks([])
            lw=1.6 if col_lbl=="HRRR Truth" else 0.6
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(lw)
            if is_std:
                stamp(ax,f"$\\bar\\sigma$={field.mean():.2g}")
            elif col_lbl!="HRRR Truth":
                stamp(ax,f"RMSE {rmse(field,tf):.2g}")
            if ri==0:
                ax.set_title(col_lbl,fontsize=7,pad=4,color=border,
                             fontweight="bold" if col_lbl=="HRRR Truth" else "normal")
            if mi==0:
                ax.set_ylabel(case_lbl,fontsize=8,labelpad=4)

    # ── Horizontal colorbars below panels ────────────────────────────────────
    # Field colorbar (wider, left portion)
    cax1=fig.add_axes([0.075, 0.12, 0.55, 0.028])
    cb1=fig.colorbar(field_ims[-1],cax=cax1,orientation="horizontal")
    cb1.set_label(unit,fontsize=7,labelpad=3)
    cb1.ax.tick_params(labelsize=6,direction="in")
    if log:
        cb1.set_ticks([0,0.1,1,5,10]); cb1.set_ticklabels(["0","0.1","1","5","10"])
    else:
        tks=neat_ticks(vmin,vmax,n=6)
        cb1.set_ticks(tks); cb1.set_ticklabels([f"{t:g}" for t in tks])

    # Std colorbar (narrower, right portion)
    cax2=fig.add_axes([0.68, 0.12, 0.27, 0.028])
    sm=plt.cm.ScalarMappable(cmap=CMAP["std"],norm=norm_std)
    cb2=fig.colorbar(sm,cax=cax2,orientation="horizontal")
    cb2.set_label(f"$\\sigma$ ({unit})",fontsize=7,labelpad=3)
    cb2.ax.tick_params(labelsize=6,direction="in")
    tks2=neat_ticks(0,max(std_max,1e-6),n=5)
    cb2.set_ticks(tks2); cb2.set_ticklabels([f"{t:g}" for t in tks2])

    fig.suptitle(f"{var['long']}",fontsize=9,fontweight="bold",y=0.965)
    proxy_footnote(fig)
    save_fig(fig,out_path)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--g1",default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--g2",default="figure_data_g2/eval_g2_corrdiff_main.nc")
    p.add_argument("--g3",default="figure_data_g3/eval_g3_main.nc")
    p.add_argument("--storm",type=int,default=2445)
    p.add_argument("--moderate",type=int,default=1842)
    p.add_argument("--calm",type=int,default=2236)
    p.add_argument("--outdir",default="paper_figures_v2")
    a=p.parse_args()
    Path(a.outdir).mkdir(parents=True,exist_ok=True)

    from fig_style import VARS_G1,VARS_G2,VARS_G3
    var_nc=[(v,a.g1) for v in VARS_G1]+[(v,a.g2) for v in VARS_G2]+\
           [(v,a.g3) for v in VARS_G3]   # both ssrd and strd
    # all 9 variables: 2t,10u,10v,sp,q,tp,sf,ssrd,strd
    keep={"2t","10u","10v","sp","q","tp","sf","ssrd","strd"}
    for v,nc in var_nc:
        if v["key"] not in keep: continue
        print(f"  Fig1 {v['key']}...")
        make_figure(v,nc,f"{a.outdir}/figure_01_{v['key']}.pdf")

if __name__=="__main__":
    main()
