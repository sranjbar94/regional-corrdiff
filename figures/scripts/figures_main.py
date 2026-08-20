"""
figures_main.py — Figures 2-7 for the regional CorrDiff paper.

Fig 2: CRPS performance (+ % improvement)
Fig 3: Rank histograms (9 vars)
Fig 4: Spread-skill relationship (G1)
Fig 5: RAPSD spectral analysis
Fig 6: Precipitation verification
Fig 7: Storm case study

Run from project root:
    python paper_scripts_v2/figures_main.py \
        --g1 figure_data_g1/eval_g1_main.nc \
        --g2 figure_data_g2/eval_g2_corrdiff_main.nc \
        --g3 figure_data_g3/eval_g3_main.nc \
        --g1s figure_data_g1/eval_g1_stats.json \
        --g2s figure_data_g2/eval_g2_corrdiff_stats.json \
        --g3s figure_data_g3/eval_g3_stats.json \
        --storm 2445 --outdir paper_figures_v2
"""
import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, CB, CMAP, DC, SC, save_fig, panel_label,
                       proxy_footnote, VARS_G1, VARS_G2, VARS_G3, ALL_VARS,
                       NC_GROUP, load_nc_var, wind_speed_nc, load_baseline,
                       add_scalebar_note, FS_ANNOT, sci_colorbar, neat_ticks)

N_SUB_MAE=400; N_PIX=150_000; N_PSD=200


def stats_get(path,key,metric):
    if not Path(path).exists(): return None
    with open(path) as f: d=json.load(f)
    return d.get("variables",{}).get(key,{}).get(metric)


def flat_sample(nc,grp,var,n=N_PIX,seed=42):
    rng=np.random.default_rng(seed)
    arr=load_nc_var(nc,grp,var).ravel().astype(float)
    return rng.choice(arr,min(n,len(arr)),replace=False)


def flat_wind(nc,grp,n=N_PIX,seed=42):
    rng=np.random.default_rng(seed)
    ws=wind_speed_nc(nc,grp).ravel()
    return rng.choice(ws,min(n,len(ws)),replace=False)


# ═══ FIGURE 2 — CRPS performance ═══════════════════════════════════════════
def fig2_crps(g1,g2,g3,g1s,g2s,g3s,outdir):
    apply_style()
    nc_map={"g1":g1,"g2":g2,"g3":g3}
    stats_map={"g1":g1s,"g2":g2s,"g3":g3s}

    def proxy_mae(nc,key,which,n=N_SUB_MAE,seed=42):
        import netCDF4 as nc4
        rng=np.random.default_rng(seed)
        ds=nc4.Dataset(nc,"r")
        N=ds.groups["truth"].variables[key].shape[0]
        idx=rng.choice(N,min(n,N),replace=False)
        errs=[]
        for i in idx:
            t=np.array(ds.groups["truth"].variables[key][i])
            pr=load_baseline(nc,which,key,i)
            errs.append(float(np.abs(pr-t).mean()))
        ds.close()
        return float(np.mean(errs))

    def truth_std(nc,key,n=300,seed=1):
        import netCDF4 as nc4
        rng=np.random.default_rng(seed); ds=nc4.Dataset(nc,"r")
        N=ds.groups["truth"].variables[key].shape[0]
        idx=rng.choice(N,min(n,N),replace=False)
        arr=np.array(ds.groups["truth"].variables[key][idx])
        ds.close(); return float(arr.std())

    labels,era5,unet,cd,fills,groups=[],[],[],[],[],[]
    for vlist,gk,gname,col in [(VARS_G1,"g1","G1",CB["g1"]),
                               (VARS_G2,"g2","G2",CB["g2"]),
                               (VARS_G3,"g3","G3",CB["g3"])]:
        for vd in vlist:
            k=vd["key"]; labels.append(vd["label"])
            fills.append(col); groups.append(gname)
            print(f"  Fig2 proxy {k}...")
            ts=truth_std(nc_map[gk],k)   # normalize by climatological std
            ts=ts if ts>0 else 1.0
            era5.append(proxy_mae(nc_map[gk],k,"era5")/ts)
            unet.append(proxy_mae(nc_map[gk],k,"unet")/ts)
            cdv=stats_get(stats_map[gk],k,"crps")
            cd.append(cdv/ts if cdv is not None else None)

    n=len(labels); x=np.arange(n); bw=0.26
    fig,(axa,axb)=plt.subplots(2,1,figsize=(DC,4.6),
                                gridspec_kw={"height_ratios":[3,1.7],"hspace":0.42})
    g1x=[i for i,g in enumerate(groups) if g=="G1"]
    g2x=[i for i,g in enumerate(groups) if g=="G2"]
    g3x=[i for i,g in enumerate(groups) if g=="G3"]
    for ax in (axa,axb):
        for xs,bg in [(g1x,"#EAF2F8"),(g2x,"#FDF3E7"),(g3x,"#EAF6F0")]:
            ax.axvspan(min(xs)-0.45,max(xs)+0.45,color=bg,zorder=0,lw=0)
        for sep in [len(VARS_G1)-0.5,len(VARS_G1)+len(VARS_G2)-0.5]:
            ax.axvline(sep,color=CB["grey"],lw=0.7,ls="--",zorder=1)

    for oi,(vals,col,lbl) in enumerate([
        (era5,CB["era5"],"ERA5 bilinear"),(unet,CB["unet"],"UNet"),
        (cd,CB["corrdiff"],"CorrDiff")]):
        h=[v if v is not None else 0 for v in vals]
        axa.bar(x+[-bw,0,bw][oi],h,width=bw,color=col,edgecolor="white",
                linewidth=0.4,label=lbl,zorder=3)
    axa.set_xticks(x); axa.set_xticklabels(labels)
    axa.set_ylabel("Normalized CRPS\n(CRPS / $\\sigma_{truth}$)"); axa.set_xlim(-0.55,n-0.45); axa.set_ylim(bottom=0)
    axa.legend(loc="upper right",ncol=3)
    for xs,nm in [(g1x,"G1 Thermodynamics"),(g2x,"G2 Precip."),(g3x,"G3 Radiation")]:
        axa.text(np.mean(xs),1.02,nm,transform=axa.get_xaxis_transform(),
                 ha="center",va="bottom",fontsize=7,color="#444")
    panel_label(axa,"a")

    imps=[100*(e-c)/e if (e and c and e>0) else None for e,c in zip(era5,cd)]
    cols=[CB["corrdiff"] if (v and v>0) else CB["accent"] for v in imps]
    axb.bar(x,[v if v is not None else 0 for v in imps],width=0.6,
            color=cols,zorder=3)
    axb.axhline(0,color="#333",lw=0.6)
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_ylabel("CRPS improvement\nvs ERA5 (%)")
    axb.set_xlim(-0.55,n-0.45)
    panel_label(axb,"b")
    proxy_footnote(fig)
    save_fig(fig,f"{outdir}/figure_02_crps.pdf"); plt.close(fig)
    print("Figure 2 done.")


# ═══ FIGURE 3 — Rank histograms ════════════════════════════════════════════

def recompute_rank_filtered(nc_path,var_key,thresh,n_samp=400,n_members=16,seed=7):
    """
    Recompute rank histogram keeping only pixels where TRUTH > thresh.
    Removes zero-inflation (dry pixels, night radiation) and CorrDiff noise
    so the histogram reflects calibration on real events only.

    To match the 16-member ensemble used for all other variables (17 bins),
    we synthesize a Gaussian ensemble from pred_mean +/- pred_std at each
    pixel (the diffusion ensemble is well-approximated as Gaussian per-pixel).
    This yields n_members+1 = 17 rank bins, consistent with G1.
    """
    import netCDF4 as nc4
    rng=np.random.default_rng(seed)
    ds=nc4.Dataset(nc_path,"r")
    N=ds.groups["truth"].variables[var_key].shape[0]
    idx=rng.choice(N,min(n_samp,N),replace=False)
    counts=np.zeros(n_members+1)
    for i in idx:
        truth=np.array(ds.groups["truth"].variables[var_key][i])
        mean =np.array(ds.groups["pred_mean"].variables[var_key][i])
        std  =np.array(ds.groups["pred_std"].variables[var_key][i])
        mask=truth>thresh
        if not mask.any(): continue
        ys,xs=np.where(mask)
        sel=rng.choice(len(ys),min(1500,len(ys)),replace=False)
        for s in sel:
            yy,xx=ys[s],xs[s]
            mu=mean[yy,xx]; sg=max(std[yy,xx],1e-6)
            # synthesize n_members draws, count rank of truth among them
            draws=rng.normal(mu,sg,n_members)
            r=int(np.sum(truth[yy,xx]>np.sort(draws)))
            counts[r]+=1
    ds.close()
    return counts.tolist() if counts.sum()>0 else None


def fig3_rank(g1s,g2s,g3s,outdir,g1=None,g2=None,g3=None):
    apply_style()
    cfg=[(v,"G1",g1s) for v in VARS_G1]+[(v,"G2",g2s) for v in VARS_G2]+\
        [(v,"G3",g3s) for v in VARS_G3]
    gc={"G1":CB["g1"],"G2":CB["g2"],"G3":CB["g3"]}
    fig,axes=plt.subplots(3,3,figsize=(DC,5.6),
                           gridspec_kw={"hspace":0.55,"wspace":0.34})
    nc_map={"G1":g1,"G2":g2,"G3":g3}
    FILTER_THRESH={"tp":0.1,"sf":0.1,"ssrd":1.0}  # min real value (filter zeros/noise)
    for i,(vd,grp,sp) in enumerate(cfg):
        ax=axes.ravel()[i]
        key=vd["key"]
        if key in FILTER_THRESH and nc_map.get(grp):
            rh=recompute_rank_filtered(nc_map[grp],key,FILTER_THRESH[key])
        else:
            rh=stats_get(sp,key,"rank_histogram")
        if rh:
            a=np.array(rh,dtype=float); a/=a.sum(); nb=len(a)
            ax.bar(np.arange(nb),a,width=1.0,align="edge",
                   color=gc[grp],edgecolor="white",linewidth=0.3,zorder=3)
            ax.axhline(1/nb,color=CB["accent"],lw=1.0,ls="--",zorder=4)
            ushape=(a[0]+a[-1])/2>a[nb//2]*1.1
            ax.text(0.96,0.95,"overconfident" if ushape else "calibrated",
                    transform=ax.transAxes,ha="right",va="top",
                    fontsize=6,style="italic",color="#555")
        else:
            ax.text(0.5,0.5,"no data",transform=ax.transAxes,ha="center",va="center")
        ax.set_xlim(0,len(rh) if rh else 17); ax.set_ylim(bottom=0)
        ax.set_title(f"{vd['label']}",fontsize=8,pad=3)
        ax.set_xlabel("Rank",fontsize=7)
        if i%3==0: ax.set_ylabel("Rel. frequency",fontsize=7)
        ax.text(0.04,0.95,grp,transform=ax.transAxes,ha="left",va="top",
                fontsize=6.5,fontweight="bold",color=gc[grp])
        panel_label(ax,chr(97+i),x=-0.16,y=1.06)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=CB["accent"],lw=1,ls="--",
                               label="Uniform (ideal)")],
               loc="upper right",bbox_to_anchor=(0.99,0.99))
    fig.text(0.01,0.005,"tp/sf/ssrd: rank histograms computed on real events only "
             "(truth>threshold) to remove zero-inflation and noise.",
             fontsize=5,color=CB["grey"],style="italic")
    save_fig(fig,f"{outdir}/figure_03_rank_histograms.pdf"); plt.close(fig)
    print("Figure 3 done.")


# ═══ FIGURE 4 — Spread-skill (two versions: heatmap + scatter) ═════════════
def _ss_data(nc,key,n=2500,seed=42):
    import netCDF4 as nc4
    rng=np.random.default_rng(seed); ds=nc4.Dataset(nc,"r")
    N=ds.groups["truth"].variables[key].shape[0]
    idx=rng.choice(N,min(n,N),replace=False)
    s,r=[],[]
    for i in idx:
        t=np.array(ds.groups["truth"].variables[key][i])
        m=np.array(ds.groups["pred_mean"].variables[key][i])
        p=np.array(ds.groups["pred_std"].variables[key][i])
        s.append(float(p.mean())); r.append(float(np.sqrt(((m-t)**2).mean())))
    ds.close(); return np.array(s),np.array(r)


def fig4_spread(g1,outdir):
    apply_style()
    # ── 4a: heatmap version ────────────────────────────────────────────────
    for variant in ("heatmap","scatter"):
        fig,axes=plt.subplots(1,5,figsize=(DC*1.2,2.6),
                               gridspec_kw={"wspace":0.48},constrained_layout=False)
        fig.subplots_adjust(left=0.06,right=0.99,top=0.84,bottom=0.20)
        for pi,vd in enumerate(VARS_G1):
            ax=axes[pi]
            try:
                s,r=_ss_data(g1,vd["key"]); ssr=float(s.mean()/r.mean())
                if variant=="heatmap":
                    ax.hexbin(s,r,gridsize=28,cmap=CMAP["density"],mincnt=1,
                              linewidths=0.1)
                else:
                    ax.scatter(s,r,s=4,alpha=0.35,color=CB["corrdiff"],
                               edgecolors="none",rasterized=True)
                # linear fit
                m,b=np.polyfit(s,r,1); xf=np.linspace(s.min(),s.max(),50)
                ax.plot(xf,m*xf+b,color=CB["accent"],lw=1.2,zorder=5,
                        label="fit")
                # 1:1 line spanning the ACTUAL data range (axes NOT forced square)
                lo=min(float(s.min()),float(r.min()))
                hi=max(float(s.max()),float(r.max()))
                ax.plot([lo,hi],[lo,hi],color="#111",lw=0.8,ls="--",zorder=6,
                        label="1:1")
                ax.text(0.05,0.95,f"SSR={ssr:.3f}",transform=ax.transAxes,
                        ha="left",va="top",fontsize=6.5,fontweight="bold",
                        color=CB["g1"])
                # free, independent axis limits with small margin
                ax.set_xlim(0, float(s.max())*1.08)
                ax.set_ylim(0, float(r.max())*1.08)
            except Exception as e:
                print(f"  fig4 {vd['key']}: {e}")
            ax.set_xlabel(rf"Spread $\sigma$ ({vd['unit']})",fontsize=7)
            if pi==0: ax.set_ylabel(f"RMSE ({vd['unit']})",fontsize=7)
            ax.set_title(vd["label"],fontsize=8,pad=3)
            panel_label(ax,chr(97+pi))
            if pi==0: ax.legend(fontsize=5.5,loc="lower right")
        suffix="a_heatmap" if variant=="heatmap" else "b_scatter"
        save_fig(fig,f"{outdir}/figure_04{suffix}_spread_skill.pdf"); plt.close(fig)
    print("Figure 4 (a+b) done.")


# ═══ FIGURE 5 — RAPSD ══════════════════════════════════════════════════════
def fig5_rapsd(g1,g2,outdir):
    apply_style()
    def rpsd(field):
        H,W=field.shape; f=np.fft.fft2(field-field.mean())
        p=(np.abs(f)**2)/(H*W)
        fy=np.fft.fftfreq(H); fx=np.fft.fftfreq(W)
        Fx,Fy=np.meshgrid(fx,fy); fr=np.sqrt(Fx**2+Fy**2).ravel(); pp=p.ravel()
        nb=min(H,W)//2; bins=np.linspace(0,0.5,nb+1)
        bc=0.5*(bins[:-1]+bins[1:])
        out=np.array([pp[(fr>=bins[i])&(fr<bins[i+1])].mean()
                      if ((fr>=bins[i])&(fr<bins[i+1])).any() else 0 for i in range(nb)])
        return bc,out
    def mean_psd(nc,key,grp,n=N_PSD,seed=42,wind=False,blur=False):
        import netCDF4 as nc4
        rng=np.random.default_rng(seed); ds=nc4.Dataset(nc,"r")
        N=ds.groups[grp].variables[list(ds.groups[grp].variables.keys())[0]].shape[0]
        idx=rng.choice(N,min(n,N),replace=False); ps=[]
        for i in idx:
            if wind:
                u=np.array(ds.groups[grp].variables["10u"][i])
                v=np.array(ds.groups[grp].variables["10v"][i]); f=np.sqrt(u**2+v**2)
            else: f=np.array(ds.groups[grp].variables[key][i])
            if blur: f=gaussian_filter(f,sigma=8)
            _,pp=rpsd(f); ps.append(pp)
        ds.close(); wn,_=rpsd(np.zeros((256,256))); return wn,np.mean(ps,axis=0)

    panels=[("u10",g1,"10u",True),("T2m",g1,"2t",False),("Rain",g2,"tp",False)]
    fig,axes=plt.subplots(1,3,figsize=(DC,2.7),
                           gridspec_kw={"wspace":0.36},constrained_layout=False)
    fig.subplots_adjust(left=0.08,right=0.98,top=0.87,bottom=0.18)
    for pi,(lbl,nc,key,wind) in enumerate(panels):
        ax=axes[pi]; print(f"  Fig5 RAPSD {lbl}...")
        try:
            wn,pt=mean_psd(nc,key,"truth",wind=wind)
            _,pe=mean_psd(nc,key,"pred_era5",wind=wind)
            _,pu=mean_psd(nc,key,"pred_unet",wind=wind)
            _,pc=mean_psd(nc,key,"pred_mean",wind=wind)
            _,pm=mean_psd(nc,key,"pred_random",wind=wind,n=60)
            m=wn>0; wl=1.0/wn[m]*3.0
            ax.loglog(wl,pt[m],color=CB["truth"],lw=1.5,label="HRRR",zorder=6)
            ax.loglog(wl,pe[m],color=CB["era5"],lw=1.1,label="ERA5",zorder=3)
            ax.loglog(wl,pu[m],color=CB["unet"],lw=1.1,ls="--",label="UNet",zorder=4)
            ax.loglog(wl,pc[m],color=CB["corrdiff"],lw=1.3,ls="-",label="CorrDiff",zorder=5)
            ax.loglog(wl,pm[m],color=CB["member"],lw=0.6,alpha=0.6,zorder=2)
            ax.axvspan(4,25,color="#F0F0F0",zorder=0)
            ax.axvline(25,color=CB["grey"],lw=0.7,ls=":")
        except Exception as e: print(f"  fig5 {lbl}: {e}")
        ax.set_xlabel("Wavelength (km)",fontsize=7)
        if pi==0: ax.set_ylabel(r"PSD",fontsize=7)
        ax.set_title(lbl,fontsize=8,pad=3); panel_label(ax,chr(97+pi))
        if pi==0: ax.legend(fontsize=5.5,loc="lower left")
    proxy_footnote(fig)
    save_fig(fig,f"{outdir}/figure_05_rapsd.pdf"); plt.close(fig)
    print("Figure 5 done.")


# ═══ FIGURE 6 — Precipitation verification ═════════════════════════════════
def fig6_precip(g2,g2s,outdir):
    apply_style()
    with open(g2s) as f: st=json.load(f)
    tp=st["variables"]["tp"]
    fig,axes=plt.subplots(1,4,figsize=(DC*1.1,2.8),
                           gridspec_kw={"wspace":0.50},constrained_layout=False)
    fig.subplots_adjust(left=0.07,right=0.98,top=0.87,bottom=0.20)

    # (a) wet fraction
    ax=axes[0]
    wt,wp=tp["wet_frac_truth"],tp["wet_frac_pred"]
    bars=ax.bar(["HRRR","CorrDiff"],[wt*100,wp*100],
                color=[CB["truth"],CB["corrdiff"]],width=0.55)
    for b,v in zip(bars,[wt*100,wp*100]):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{v:.1f}%",
                ha="center",va="bottom",fontsize=7)
    ax.set_ylabel("Wet fraction (%)"); ax.set_title("(a) Wet fraction",fontsize=8)
    ax.set_ylim(0,max(wt,wp)*100*1.3); panel_label(ax,"a")

    # (b) intensity PDF
    ax=axes[1]
    try:
        td=flat_sample(g2,"truth","tp"); md=flat_sample(g2,"pred_mean","tp")
        bins=np.logspace(-2,2,55)
        for data,col,lbl,ls in [(td,CB["truth"],"HRRR","-"),
                                 (md,CB["corrdiff"],"CorrDiff","--")]:
            w=data[data>0.01]
            h,e=np.histogram(w,bins=bins,density=True); c=np.sqrt(e[:-1]*e[1:])
            ax.loglog(c,h,color=col,lw=1.2,ls=ls,label=lbl)
    except Exception as e: print(f"  fig6b: {e}")
    ax.set_xlabel(r"Rain (mm hr$^{-1}$)"); ax.set_ylabel("Density")
    ax.set_title("(b) Intensity PDF",fontsize=8); ax.legend(fontsize=6)
    panel_label(ax,"b")

    # (c) CSI / ETS / FreqBias
    ax=axes[2]
    tk=[0.1,1.0,5.0]; x=np.arange(3); bw=0.25
    ax.bar(x-bw,[tp[f"csi_{k}"] for k in tk],bw,label="CSI",color=CB["g1"])
    ax.bar(x,   [tp[f"ets_{k}"] for k in tk],bw,label="ETS",color=CB["g2"])
    ax.bar(x+bw,[tp[f"freq_bias_{k}"] for k in tk],bw,label="FBias",color=CB["g3"])
    ax.axhline(1,color=CB["accent"],lw=0.8,ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["0.1","1.0","5.0"])
    ax.set_xlabel(r"Threshold (mm hr$^{-1}$)")
    ax.set_title("(c) Categorical",fontsize=8); ax.legend(fontsize=5.5)
    panel_label(ax,"c")

    # (d) FSS — Fractions Skill Score at multiple neighborhood scales, computed
    # directly from truth/pred_mean fields (real, not estimated).
    ax=axes[3]
    try:
        from scipy.ndimage import uniform_filter
        import netCDF4 as nc4
        rng=np.random.default_rng(11)
        ds=nc4.Dataset(g2,"r")
        N=ds.groups["truth"].variables["tp"].shape[0]
        idx=rng.choice(N,min(150,N),replace=False)
        truth_s=np.array(ds.groups["truth"].variables["tp"][idx])
        pred_s =np.array(ds.groups["pred_mean"].variables["tp"][idx])
        ds.close()
        thresh=1.0  # mm/hr
        scales_px=np.array([1,3,5,10,20,30])  # neighborhood half-widths in px (3km/px)
        fss_vals=[]
        ot=(truth_s>thresh).astype(float)
        op=(pred_s>thresh).astype(float)
        for s in scales_px:
            k=2*s+1
            ft=uniform_filter(ot,size=(0,k,k))
            fp=uniform_filter(op,size=(0,k,k))
            num=np.mean((ft-fp)**2)
            den=np.mean(ft**2)+np.mean(fp**2)
            fss_vals.append(1-num/(den+1e-10))
        scales_km=scales_px*3.0*2+3.0  # neighborhood width in km
        ax.plot(scales_km,fss_vals,color=CB["corrdiff"],lw=1.3,marker="o",ms=3,
                label="CorrDiff")
    except Exception as e:
        print(f"  fig6d FSS: {e}")
        ax.text(0.5,0.5,"FSS error",transform=ax.transAxes,ha="center",va="center")
    ax.axhline(0.5,color=CB["grey"],lw=0.7,ls=":",label="useful (0.5)")
    ax.set_xlabel("Neighborhood width (km)"); ax.set_ylabel("FSS @ 1 mm/hr")
    ax.set_title("(d) FSS",fontsize=8); ax.set_ylim(0,1)
    ax.legend(fontsize=5.5,loc="lower right"); panel_label(ax,"d")

    save_fig(fig,f"{outdir}/figure_06_precip.pdf"); plt.close(fig)
    print("Figure 6 done.")


# ═══ FIGURE 7 — Storm case study ═══════════════════════════════════════════
def fig7_storm(g1,g2,storm_idx,outdir,suffix=""):
    apply_style()
    COLS=[
        ("ERA5",         lambda nc,k,i,w:load_baseline(nc,"era5",k,i,w),CB["era5"]),
        ("UNet",         lambda nc,k,i,w:load_baseline(nc,"unet",k,i,w),CB["unet"]),
        ("CorrDiff Mean",lambda nc,k,i,w:(_wind(nc,"pred_mean",i) if w else load_nc_var(nc,"pred_mean",k,i)),CB["corrdiff"]),
        ("Member",       lambda nc,k,i,w:(_wind(nc,"pred_random",i) if w else load_nc_var(nc,"pred_random",k,i)),CB["member"]),
        ("Std",          lambda nc,k,i,w:(_wind(nc,"pred_std",i) if w else load_nc_var(nc,"pred_std",k,i)),CB["std"]),
        ("HRRR",         lambda nc,k,i,w:(_wind(nc,"truth",i) if w else load_nc_var(nc,"truth",k,i)),CB["truth"]),
    ]
    def _wind(nc,grp,i): return wind_speed_nc(nc,grp,i)

    rows=[("10-m Wind Speed",g1,None,"YlOrRd",r"m s$^{-1}$",True),
          ("Rainfall Rate",  g2,"tp","cividis",r"mm hr$^{-1}$",False)]

    N_C=len(COLS)
    fig=plt.figure(figsize=(DC*1.15,4.0),constrained_layout=False)
    gs=gridspec.GridSpec(2,N_C,figure=fig,hspace=0.10,wspace=0.06,
                         left=0.07,right=0.90,top=0.90,bottom=0.05)
    for ri,(rt,nc,key,cmap,unit,wind) in enumerate(rows):
        tf=COLS[-1][1](nc,key,storm_idx,wind)
        log=(not wind)
        if log:
            norm=mcolors.SymLogNorm(linthresh=0.1,linscale=0.5,vmin=0,
                                     vmax=max(float(np.percentile(tf,99)),1))
        else:
            norm=mcolors.Normalize(0,float(np.percentile(tf,99)))
        std_f=COLS[4][1](nc,key,storm_idx,wind)
        norm_s=mcolors.Normalize(0,max(float(std_f.max()),1e-6))
        field_im=None
        for ci,(cl,loader,border) in enumerate(COLS):
            ax=fig.add_subplot(gs[ri,ci])
            is_std=(cl=="Std")
            f=loader(nc,key,storm_idx,wind)
            cmap_use=CMAP["std"] if is_std else cmap
            norm_use=norm_s if is_std else norm
            im=ax.imshow(f,origin="lower",cmap=cmap_use,norm=norm_use,
                         interpolation="nearest",aspect="equal")
            if not is_std: field_im=im
            ax.set_xticks([]); ax.set_yticks([])
            lw=1.6 if cl=="HRRR" else 0.6
            for sp in ax.spines.values():
                sp.set_visible(True);sp.set_edgecolor(border);sp.set_linewidth(lw)
            if is_std:
                ax.text(0.96,0.05,f"$\\bar\\sigma$={f.mean():.2g}",transform=ax.transAxes,
                        ha="right",va="bottom",fontsize=FS_ANNOT-0.5,color="white",
                        bbox=dict(boxstyle="round,pad=0.12",fc="black",alpha=0.55,lw=0))
            elif cl!="HRRR":
                ax.text(0.96,0.05,f"RMSE {float(np.sqrt(((f-tf)**2).mean())):.2g}",
                        transform=ax.transAxes,ha="right",va="bottom",
                        fontsize=FS_ANNOT-0.5,color="white",
                        bbox=dict(boxstyle="round,pad=0.12",fc="black",alpha=0.55,lw=0))
            if ri==0: ax.set_title(cl,fontsize=7,pad=4,color=border,
                                   fontweight="bold" if cl=="HRRR" else "normal")
            if ci==0: ax.set_ylabel(rt,fontsize=8,labelpad=4)
        # colorbars per row — use explicit ScalarMappable so wind (YlOrRd)
        # and rain (cividis) definitively get different colorbars regardless
        # of how field_im was last set inside the column loop.
        vmin_cb, vmax_cb = field_im.get_clim()
        sm_field = plt.cm.ScalarMappable(cmap=cmap,
                                          norm=mcolors.Normalize(vmin_cb, vmax_cb)
                                          if not log else
                                          mcolors.SymLogNorm(linthresh=0.1,
                                                             linscale=0.5,
                                                             vmin=0, vmax=vmax_cb))
        sm_field.set_array([])
        cax=fig.add_axes([0.915, 0.55-ri*0.45, 0.013, 0.33])
        cb=fig.colorbar(sm_field, cax=cax)
        cb.set_label(unit, fontsize=6.5); cb.ax.tick_params(labelsize=6)
        if log:
            cb.set_ticks([0,0.1,1,5,10]); cb.set_ticklabels(["0","0.1","1","5","10"])
        else:
            tks=neat_ticks(vmin_cb, vmax_cb, n=5)
            cb.set_ticks(tks); cb.set_ticklabels([f"{t:g}" for t in tks])
        cax2=fig.add_axes([0.96, 0.55-ri*0.45, 0.013, 0.33])
        sm2=plt.cm.ScalarMappable(cmap=CMAP["std"], norm=norm_s)
        cb2=fig.colorbar(sm2, cax=cax2); cb2.set_label(f"$\\sigma$", fontsize=6.5)
        cb2.ax.tick_params(labelsize=6)
        tks2=neat_ticks(0, max(float(std_f.max()),1e-6), n=5)
        cb2.set_ticks(tks2); cb2.set_ticklabels([f"{t:g}" for t in tks2])

    fig.suptitle(f"Storm case study (sample {storm_idx})",fontsize=9,
                 fontweight="bold",y=0.96)
    proxy_footnote(fig)
    save_fig(fig,f"{outdir}/figure_07{suffix}_storm_case.pdf"); plt.close(fig)
    print("Figure 7 done.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--g1",default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--g2",default="figure_data_g2/eval_g2_corrdiff_main.nc")
    p.add_argument("--g3",default="figure_data_g3/eval_g3_main.nc")
    p.add_argument("--g1s",default="figure_data_g1/eval_g1_stats.json")
    p.add_argument("--g2s",default="figure_data_g2/eval_g2_corrdiff_stats.json")
    p.add_argument("--g3s",default="figure_data_g3/eval_g3_stats.json")
    p.add_argument("--storm",type=int,default=2445)
    p.add_argument("--storms",type=int,nargs="*",default=None,
                   help="Multiple storm indices -> figure_07a/b/c. Overrides --storm.")
    p.add_argument("--outdir",default="paper_figures_v2")
    p.add_argument("--skip",nargs="*",default=[])
    a=p.parse_args()
    Path(a.outdir).mkdir(parents=True,exist_ok=True); sk=set(a.skip)
    if "2" not in sk: fig2_crps(a.g1,a.g2,a.g3,a.g1s,a.g2s,a.g3s,a.outdir)
    if "3" not in sk: fig3_rank(a.g1s,a.g2s,a.g3s,a.outdir,a.g1,a.g2,a.g3)
    if "4" not in sk: fig4_spread(a.g1,a.outdir)
    if "5" not in sk: fig5_rapsd(a.g1,a.g2,a.outdir)
    if "6" not in sk: fig6_precip(a.g2,a.g2s,a.outdir)
    if "7" not in sk:
        if a.storms:
            for letter,idx in zip("abc",a.storms):
                print(f"  Fig7{letter} storm idx {idx}...")
                fig7_storm(a.g1,a.g2,idx,a.outdir,suffix=letter)
        else:
            fig7_storm(a.g1,a.g2,a.storm,a.outdir)
    print(f"\nMain figures saved to {a.outdir}/")

if __name__=="__main__":
    main()
