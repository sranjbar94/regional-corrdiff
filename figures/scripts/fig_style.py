"""
fig_style.py — Publication style for "Kilometer-Scale Atmospheric Downscaling
at Coastal Ocean Areas using Residual Corrective Diffusion Modeling"

Target journal : AGU (JGR Machine Learning & Computation)
Column widths  : single 3.5 in | double 7.2 in
DPI            : 600 (high-impact submission)
Font           : Helvetica/Arial sans, 7-9 pt
Colormaps      : perceptually-uniform, colorblind-safe (viridis family)

Baseline data:
  pred_era5 — ERA5 bilinear interpolation (32x32 → 256x256, already in eval files)
  pred_unet — UNet regression-only inference (Stage 1, no diffusion)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Baseline configuration (REAL DATA — no proxies) ──────────────────────────
BASELINE_GROUP_ERA5     = "pred_era5"
BASELINE_GROUP_UNET     = "pred_unet"
BASELINE_ERA5_BLUR      = False
BASELINE_PROXIES_ACTIVE = False

# ── Page geometry (AGU) ───────────────────────────────────────────────────────
SC  = 3.5
DC  = 7.2
DPI = 600

# ── Typography sizes (pt) ─────────────────────────────────────────────────────
FS_TITLE  = 9
FS_LABEL  = 8
FS_TICK   = 7
FS_ANNOT  = 6.5
FS_PANEL  = 9
FS_LEGEND = 7

# ── Colorblind-safe palette (Wong/Tol) ────────────────────────────────────────
CB = {
    "era5":     "#0072B2",
    "unet":     "#E69F00",
    "corrdiff": "#009E73",
    "member":   "#56B4E9",
    "truth":    "#000000",
    "std":      "#CC79A7",
    "g1":       "#0072B2",
    "g2":       "#E69F00",
    "g3":       "#009E73",
    "accent":   "#D55E00",
    "grey":     "#999999",
    "grey_lt":  "#CCCCCC",
}
COLORS = CB

CMAP = {
    "temp":"viridis","wind":"cividis","wind_div":"RdBu_r","pressure":"viridis",
    "humidity":"viridis","precip":"cividis","radiation":"viridis",
    "std":"magma","diff":"RdBu_r","corr":"RdBu_r","density":"viridis",
}

# ── Variable metadata ─────────────────────────────────────────────────────────
VARS_G1 = [
    {"key":"2t",  "label":"T2m",   "unit":"K",             "long":"2-m Temperature",     "cmap":"viridis","log":False},
    {"key":"10u", "label":"u10",   "unit":r"m s$^{-1}$",  "long":"10-m Zonal Wind",     "cmap":"RdBu_r", "log":False},
    {"key":"10v", "label":"v10",   "unit":r"m s$^{-1}$",  "long":"10-m Meridional Wind","cmap":"RdBu_r", "log":False},
    {"key":"sp",  "label":"$p_s$", "unit":"Pa",            "long":"Surface Pressure",    "cmap":"viridis","log":False},
    {"key":"q",   "label":"q",     "unit":r"kg kg$^{-1}$","long":"Specific Humidity",   "cmap":"viridis","log":False},
]
VARS_G2 = [
    {"key":"tp",  "label":"Rain",  "unit":r"mm hr$^{-1}$","long":"Rainfall Rate","cmap":"cividis","log":True},
    {"key":"sf",  "label":"Snow",  "unit":r"mm hr$^{-1}$","long":"Snowfall Rate","cmap":"cividis","log":True},
]
VARS_G3 = [
    {"key":"ssrd","label":r"SW$\downarrow$","unit":r"W m$^{-2}$","long":"Shortwave Radiation","cmap":"viridis","log":False},
    {"key":"strd","label":r"LW$\downarrow$","unit":r"W m$^{-2}$","long":"Longwave Radiation", "cmap":"viridis","log":False},
]
ALL_VARS = VARS_G1 + VARS_G2 + VARS_G3
NC_GROUP = {**{v["key"]:"g1" for v in VARS_G1},
            **{v["key"]:"g2" for v in VARS_G2},
            **{v["key"]:"g3" for v in VARS_G3}}


def apply_style():
    mpl.rcParams.update({
        "font.family":"sans-serif",
        "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
        "mathtext.fontset":"dejavusans",
        "font.size":FS_LABEL,"axes.titlesize":FS_TITLE,"axes.labelsize":FS_LABEL,
        "xtick.labelsize":FS_TICK,"ytick.labelsize":FS_TICK,
        "legend.fontsize":FS_LEGEND,"figure.titlesize":FS_TITLE,
        "axes.linewidth":0.7,
        "xtick.major.width":0.7,"ytick.major.width":0.7,
        "xtick.minor.width":0.5,"ytick.minor.width":0.5,
        "xtick.major.size":3.0,"ytick.major.size":3.0,
        "xtick.minor.size":1.5,"ytick.minor.size":1.5,
        "xtick.direction":"out","ytick.direction":"out",
        "lines.linewidth":1.2,"patch.linewidth":0.6,
        "figure.dpi":150,"savefig.dpi":DPI,
        "savefig.bbox":"tight","savefig.pad_inches":0.02,
        "pdf.fonttype":42,"ps.fonttype":42,
        "legend.frameon":True,"legend.framealpha":0.9,
        "legend.edgecolor":CB["grey_lt"],"legend.handlelength":1.5,
        "legend.borderpad":0.4,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.axisbelow":True,
    })


def panel_label(ax, letter, x=-0.16, y=1.02, **kw):
    ax.text(x,y,f"({letter})",transform=ax.transAxes,
            fontsize=FS_PANEL,fontweight="bold",va="bottom",ha="left",**kw)


def save_fig(fig, path):
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=DPI,bbox_inches="tight")
    png=str(path).rsplit(".",1)[0]+".png"
    fig.savefig(png,dpi=300,bbox_inches="tight")
    print(f"  Saved -> {path}  (+ .png preview)")


def proxy_footnote(fig, y=0.002):
    """No-op — real baselines are in use, no proxy footnote needed."""
    pass


def load_nc_var(nc_path, group, var, idx=None):
    import netCDF4 as nc4
    ds=nc4.Dataset(nc_path,"r")
    v=ds.groups[group].variables[var]
    data=np.array(v[:] if idx is None else v[idx])
    ds.close()
    return data


def load_baseline(nc_path, which, var, idx=None, wind=False):
    grp=BASELINE_GROUP_ERA5 if which=="era5" else BASELINE_GROUP_UNET
    if wind:
        u=load_nc_var(nc_path,grp,"10u",idx); v=load_nc_var(nc_path,grp,"10v",idx)
        field=np.sqrt(u**2+v**2)
    else:
        field=load_nc_var(nc_path,grp,var,idx)
    # No blurring needed — pred_era5 already contains real bilinear-interpolated ERA5
    return field


def wind_speed_nc(nc_path, group, idx=None):
    u=load_nc_var(nc_path,group,"10u",idx); v=load_nc_var(nc_path,group,"10v",idx)
    return np.sqrt(u**2+v**2)


def add_scalebar_note(ax, text="3 km / px", loc="lower left"):
    xy={"lower left":(0.03,0.04),"lower right":(0.97,0.04)}[loc]
    ha="left" if "left" in loc else "right"
    ax.text(*xy,text,transform=ax.transAxes,ha=ha,va="bottom",
            fontsize=5.5,color="white",
            bbox=dict(boxstyle="round,pad=0.1",fc="black",alpha=0.45,lw=0))


def neat_ticks(vmin, vmax, n=5):
    """
    Return n evenly-spaced tick values between vmin and vmax, each rounded
    to a "neat" number (1, 2, 5, 10, 25, 50, 100, 0.25, 0.5, 0.75, 0.1
    etc.) so colorbars show tidy values like 0, 5, 10, 15, 20 rather than
    arbitrary decimals like 1.24 or 7.83.

    Strategy: compute the natural spacing (vmax-vmin)/(n-1), then round it
    UP to the nearest value from the canonical neat-step sequence
    {1,2,2.5,5} × 10^k, which gives the same "nice" rounding that most
    commercial plotting packages use for auto-ticks. The final tick array
    is then anchored to a clean multiple of that step that is >= vmin, and
    trimmed/extended so the first and last ticks are within [vmin, vmax].
    """
    if vmax <= vmin:
        return np.array([vmin, vmax])
    raw_step = (vmax - vmin) / max(n - 1, 1)
    # Find the magnitude (power of 10) of the raw step
    mag = 10.0 ** np.floor(np.log10(raw_step))
    # Normalised step in [1, 10)
    norm = raw_step / mag
    # Round up to nearest canonical multiplier
    for mult in [1, 2, 2.5, 5, 10]:
        if mult >= norm:
            step = mult * mag
            break
    else:
        step = 10 * mag

    # Anchor: first tick >= vmin that is a clean multiple of step
    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 0.01, step)
    # Keep only values strictly within (or very close to) [vmin, vmax]
    ticks = ticks[(ticks >= vmin - step * 0.01) & (ticks <= vmax + step * 0.01)]
    # Always include the endpoints rounded to the step
    return np.unique(np.round(ticks / step) * step)


def sci_colorbar(cb, unit_label, fontsize_label=7, fontsize_tick=6,
                  n_ticks=3, labelpad=3):
    """
    Format a colorbar with exactly `n_ticks` ticks, each shown as a
    2-decimal mantissa in [-10,10], with the shared power-of-10 factor
    folded into the axis label as "(x10^n)". Keeps tick labels short so
    adjacent colorbars don't overlap.
    """
    vmin, vmax = cb.mappable.get_clim()
    if vmin is None or vmax is None or not np.isfinite([vmin, vmax]).all():
        cb.ax.tick_params(labelsize=fontsize_tick)
        cb.set_label(unit_label, fontsize=fontsize_label, labelpad=labelpad)
        return unit_label

    ticks = np.linspace(vmin, vmax, n_ticks)
    absmax = max(abs(vmin), abs(vmax))
    if absmax == 0 or not np.isfinite(absmax):
        exp = 0
    else:
        exp = int(np.floor(np.log10(absmax)))
        if absmax / (10.0**exp) >= 10:
            exp += 1

    factor = 10.0**exp
    mantissas = ticks / factor
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{m:.2f}" for m in mantissas])
    cb.ax.tick_params(labelsize=fontsize_tick)

    if exp == 0:
        label = unit_label
    else:
        label = f"{unit_label}\n" + r"($\times10^{%d}$)" % exp
    cb.set_label(label, fontsize=fontsize_label, labelpad=labelpad)
    return label
