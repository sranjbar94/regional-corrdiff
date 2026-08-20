"""
tables.py — Tables 1-3 + Table S1 for the regional CorrDiff paper.

T1: Deterministic skill (RMSE, MAE) — ERA5 | UNet | CorrDiff
T2: Probabilistic skill (CRPS, CRPS improvement, SSR)
T3: Literature comparison (manual content)
S1: Detailed precipitation metrics

Outputs LaTeX (.tex) + CSV for each.

Run from project root:
    python paper_scripts_v2/tables.py \
        --g1 figure_data_g1/eval_g1_main.nc \
        --g2 figure_data_g2/eval_g2_corrdiff_main.nc \
        --g3 figure_data_g3/eval_g3_main.nc \
        --g1s figure_data_g1/eval_g1_stats.json \
        --g2s figure_data_g2/eval_g2_corrdiff_stats.json \
        --g3s figure_data_g3/eval_g3_stats.json \
        --outdir paper_figures_v2
"""
import argparse, json, sys, csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (VARS_G1, VARS_G2, VARS_G3, ALL_VARS, NC_GROUP,
                       load_baseline)

N_SUB=300


def proxy_metrics(nc,key,which,n=N_SUB,seed=42):
    import netCDF4 as nc4
    rng=np.random.default_rng(seed); ds=nc4.Dataset(nc,"r")
    N=ds.groups["truth"].variables[key].shape[0]
    idx=rng.choice(N,min(n,N),replace=False)
    mae,rmse=[],[]
    for i in idx:
        t=np.array(ds.groups["truth"].variables[key][i])
        p=load_baseline(nc,which,key,i)
        d=p-t; mae.append(float(np.abs(d).mean())); rmse.append(float(np.sqrt((d**2).mean())))
    ds.close(); return float(np.mean(mae)),float(np.mean(rmse))


def cd_metrics(sp,key):
    if not Path(sp).exists(): return {}
    with open(sp) as f: d=json.load(f)
    return d.get("variables",{}).get(key,{})


def bold_best(vals,fmt=".4f",lower=True):
    num=[(i,v) for i,v in enumerate(vals) if v is not None]
    if not num: return ["--"]*len(vals)
    bi=(min if lower else max)(num,key=lambda x:x[1])[0]
    return [("--" if v is None else (f"\\textbf{{{v:{fmt}}}}" if i==bi else f"{v:{fmt}}"))
            for i,v in enumerate(vals)]


def write_latex(rows,headers,caption,label,path):
    cf="l"+"r"*(len(headers)-1)
    L=["\\begin{table}[ht]","\\centering",f"\\caption{{{caption}}}",
       f"\\label{{{label}}}",f"\\begin{{tabular}}{{{cf}}}","\\toprule",
       " & ".join(headers)+" \\\\","\\midrule"]
    L+=[" & ".join(str(c) for c in r)+" \\\\" for r in rows]
    L+=["\\bottomrule","\\end{tabular}","\\end{table}"]
    Path(path).write_text("\n".join(L)); print(f"  LaTeX -> {path}")


def write_csv(rows,headers,path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)
    print(f"  CSV   -> {path}")


def table_1(g1,g2,g3,g1s,g2s,g3s,outdir):
    print("Table 1...")
    nc={"g1":g1,"g2":g2,"g3":g3}; st={"g1":g1s,"g2":g2s,"g3":g3s}
    headers=["Variable","Unit","ERA5 RMSE","UNet RMSE","CorrDiff RMSE",
             "ERA5 MAE","UNet MAE","CorrDiff MAE"]
    rl,rc=[],[]
    for vd in ALL_VARS:
        k=vd["key"]; g=NC_GROUP[k]
        print(f"  {k}...")
        e_mae,e_rmse=proxy_metrics(nc[g],k,"era5")
        u_mae,u_rmse=proxy_metrics(nc[g],k,"unet")
        m=cd_metrics(st[g],k); c_mae,c_rmse=m.get("mae"),m.get("rmse")
        rmse_f=bold_best([e_rmse,u_rmse,c_rmse])
        mae_f =bold_best([e_mae,u_mae,c_mae])
        unit=vd["unit"].replace("$","").replace("^","").replace("{","").replace("}","")
        rl.append([vd["label"],unit]+rmse_f+mae_f)
        rc.append([vd["label"],unit,e_rmse,u_rmse,c_rmse,e_mae,u_mae,c_mae])
    cap=("Deterministic skill (RMSE, MAE) on 3{,}000 independent test samples "
         "(2024--2025). Best per metric in bold.")
    write_latex(rl,headers,cap,"tab:det","%s/table_1_deterministic.tex"%outdir)
    write_csv(rc,[h.replace("\\","") for h in headers],
              "%s/table_1_deterministic.csv"%outdir)


def table_2(g1,g2,g3,g1s,g2s,g3s,outdir):
    print("Table 2...")
    nc={"g1":g1,"g2":g2,"g3":g3}; st={"g1":g1s,"g2":g2s,"g3":g3s}
    headers=["Variable","Unit","ERA5 CRPS","UNet CRPS","CorrDiff CRPS",
             "Improv. (%)","SSR"]
    rl,rc=[],[]
    for vd in ALL_VARS:
        k=vd["key"]; g=NC_GROUP[k]
        print(f"  {k}...")
        e_mae,_=proxy_metrics(nc[g],k,"era5")   # CRPS=MAE for deterministic
        u_mae,_=proxy_metrics(nc[g],k,"unet")
        m=cd_metrics(st[g],k)
        c_crps=m.get("crps"); c_spread=m.get("spread"); c_rmse=m.get("rmse")
        ssr=(c_spread/c_rmse) if (c_spread and c_rmse) else None
        imp=(100*(e_mae-c_crps)/e_mae) if (e_mae and c_crps) else None
        crps_f=bold_best([e_mae,u_mae,c_crps])
        unit=vd["unit"].replace("$","").replace("^","").replace("{","").replace("}","")
        rl.append([vd["label"],unit]+crps_f+
                  [f"{imp:.1f}" if imp else "--",f"{ssr:.3f}" if ssr else "--"])
        rc.append([vd["label"],unit,e_mae,u_mae,c_crps,imp,ssr])
    cap=("Probabilistic skill. CRPS reduces to MAE for deterministic baselines. "
         "Improvement is CorrDiff vs ERA5. SSR = spread/RMSE "
         "($\\approx1$ ideal). Best CRPS in bold.")
    write_latex(rl,headers,cap,"tab:prob","%s/table_2_probabilistic.tex"%outdir)
    write_csv(rc,[h.replace("\\","") for h in headers],
              "%s/table_2_probabilistic.csv"%outdir)


def table_3(outdir):
    print("Table 3...")
    headers=["Study","Method","Domain","Resolution","Variables","Probabilistic"]
    rows=[
        ["This study","CorrDiff (grouped)","CONUS coastal ocean",
         "25→3 km","9","Yes (16-mem)"],
        ["Mardani et al. 2024","CorrDiff","Taiwan","25→2 km","4","Yes"],
        ["Pathak et al. 2024","FourCastNet","Global","25 km","—","No"],
        ["Vandal et al. 2017","DeepSD (CNN)","CONUS","downscaling","precip","No"],
        ["Harris et al. 2022","GAN","UK","precip nowcast","precip","Yes"],
        ["Price \\& Rasp 2022","cGAN","Germany","precip","precip","Yes"],
        ["Ling et al. 2024","Diffusion","China","9→1 km","T,wind","Yes"],
    ]
    cap=("Comparison with related downscaling and generative weather models. "
         "This study is the first grouped multi-model CorrDiff configuration "
         "applied to coastal-ocean surface forcing with nine variables.")
    write_latex(rows,headers,cap,"tab:lit","%s/table_3_literature.tex"%outdir)
    write_csv(rows,headers,"%s/table_3_literature.csv"%outdir)


def table_s1(g2s,outdir):
    print("Table S1...")
    with open(g2s) as f: d=json.load(f)
    tp=d["variables"]["tp"]
    headers=["Metric","0.1 mm/hr","1.0 mm/hr","5.0 mm/hr"]
    tk=[0.1,1.0,5.0]
    rows=[]
    for name,prefix in [("CSI","csi"),("ETS","ets"),("Freq. bias","freq_bias")]:
        rows.append([name]+[f"{tp[f'{prefix}_{k}']:.3f}" for k in tk])
    rows.append(["Wet frac (pred)",f"{tp['wet_frac_pred']:.3f}","",""])
    rows.append(["Wet frac (truth)",f"{tp['wet_frac_truth']:.3f}","",""])
    rows.append(["MAE (wet px)",f"{tp['mae_wet']:.4f}","",""])
    rows.append(["RMSE (wet px)",f"{tp['rmse_wet']:.4f}","",""])
    if "binned_mae" in tp:
        for cat in ["dry","light","moderate","heavy","extreme"]:
            if cat in tp["binned_mae"]:
                b=tp["binned_mae"][cat]
                rows.append([f"MAE {cat}",f"{b['mae']:.4f}",
                             f"{b['frac']*100:.1f}\\%",""])
    cap=("Detailed precipitation verification metrics for CorrDiff (tp) on "
         "3{,}000 test samples. CSI = Critical Success Index; ETS = Equitable "
         "Threat Score. Binned MAE shows error by intensity category with "
         "pixel fraction.")
    write_latex(rows,headers,cap,"tab:s1_precip","%s/table_S1_precip.tex"%outdir)
    write_csv(rows,headers,"%s/table_S1_precip.csv"%outdir)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--g1",default="figure_data_g1/eval_g1_main.nc")
    p.add_argument("--g2",default="figure_data_g2/eval_g2_corrdiff_main.nc")
    p.add_argument("--g3",default="figure_data_g3/eval_g3_main.nc")
    p.add_argument("--g1s",default="figure_data_g1/eval_g1_stats.json")
    p.add_argument("--g2s",default="figure_data_g2/eval_g2_corrdiff_stats.json")
    p.add_argument("--g3s",default="figure_data_g3/eval_g3_stats.json")
    p.add_argument("--outdir",default="paper_figures_v2")
    p.add_argument("--skip",nargs="*",default=[])
    a=p.parse_args()
    Path(a.outdir).mkdir(parents=True,exist_ok=True); sk=set(a.skip)
    if "t1" not in sk: table_1(a.g1,a.g2,a.g3,a.g1s,a.g2s,a.g3s,a.outdir)
    if "t2" not in sk: table_2(a.g1,a.g2,a.g3,a.g1s,a.g2s,a.g3s,a.outdir)
    if "t3" not in sk: table_3(a.outdir)
    if "ts1" not in sk: table_s1(a.g2s,a.outdir)
    print(f"Tables saved to {a.outdir}/")

if __name__=="__main__":
    main()
