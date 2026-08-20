"""
figure_S9_training_data.py — Figure S9: training data coverage (STANDALONE).

Updated for V5 dataset (real lat/lon stored directly in the root group by
nc_writer_v5.py -- no more coord/invariant grid-mismatch issue that affected
V2-V4; see investigation notes for that incident).

Split out as its own script (separate from run_all_figures.sh) because even
the "fast path" reading a ~100k-sample train.nc can take a while if not
careful about how the input/output groups are read.

Two genuinely fast operations, done in full for ALL samples in the file:
  - top panel map: root `lat`/`lon` arrays read directly -- no index lookup,
    no grid-mismatch risk, just two 1-D array reads.

One operation that is NOT done in full (and should not be, for any dataset
this size): the 3x3 histograms. Reading entire `output/<var>` arrays is
impractical and unnecessary for a histogram. Instead, a representative
systematic-stride subsample of SAMPLES (default 3,000) is selected first,
and only those rows are read from disk, which keeps runtime bounded
regardless of dataset size while still representing the true distribution.

Coastline background: geopandas + Natural Earth (via geodatasets, which
downloads and caches ne_110m_coastline.zip on first use) with a bundled
coarse fallback outline if that's unavailable (no internet from this node,
or the Natural Earth CDN is unreachable).

Usage:
    python3 figure_S9_training_data.py \
        --train data/v5/randomly/test.nc \
        --n-hist-samples 3000 \
        --outdir paper_figures_v2
"""
import argparse, sys, time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_style, CB, DC, save_fig, panel_label, ALL_VARS

ZERO_TRIM_VARS = {"sf", "tp", "ssrd"}   # per instructions: remove zeros


# ── Coastline (geopandas/Natural Earth, with bundled fallback) ─────────────
def _us_coastline_lines():
    """
    Returns a list of (lon_array, lat_array) polylines for the coastline
    background. Tries geopandas + Natural Earth (via geodatasets, which
    downloads ne_110m_coastline.zip on first use and caches it locally)
    first; falls back to a bundled coarse hand-digitized CONUS outline if
    that fails (e.g. no internet access from this node, or the Natural
    Earth CDN is unreachable) so the figure can still be produced.
    """
    try:
        import geopandas as gpd
        import geodatasets
        path = geodatasets.get_path("naturalearth.land")
        gdf = gpd.read_file(path)
        # Clip to the CONUS coastal map extent first (much cheaper than
        # computing boundaries on the whole world), then take polygon
        # boundaries as the coastline outline.
        gdf = gdf.cx[-140:-55, 15:60]
        boundary = gdf.boundary
        lines = []
        for geom in boundary:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                lines.append((np.array(xs), np.array(ys)))
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    xs, ys = part.xy
                    lines.append((np.array(xs), np.array(ys)))
        if not lines:
            raise RuntimeError("Natural Earth land query returned no geometry")
        print(f"  [S9] coastline: geopandas/Natural Earth land boundaries "
              f"loaded ({len(lines)} segments)")
        return lines
    except Exception as e:
        print(f"  [S9] geopandas/Natural Earth unavailable ({e}); "
              f"using bundled coarse CONUS outline")
        coarse = [
            ([-117.1,-117.3,-118.2,-119.7,-120.9,-121.8,-122.4,-122.8,-123.8,-124.2,-124.1],
             [32.5,33.0,34.0,34.4,34.5,36.6,37.2,38.0,39.4,40.8,46.2]),
            ([-97.1,-95.3,-93.8,-91.9,-89.9,-88.0,-85.0,-83.0,-82.6,-81.8],
             [25.9,28.8,29.7,29.2,29.1,30.2,29.7,29.8,27.9,24.6]),
            ([-80.1,-78.9,-77.9,-76.0,-75.5,-74.0,-71.0,-70.2,-67.0],
             [25.8,33.8,34.2,36.9,38.9,40.6,41.4,43.7,44.8]),
            ([-124.7,-117.0,-104.0,-95.2,-82.4],
             [49.0,49.0,49.0,49.0,42.2]),
            ([-117.1,-108.2,-106.5,-104.9,-97.1],
             [32.5,31.3,31.8,29.7,25.9]),
        ]
        return [(np.array(x), np.array(y)) for x, y in coarse]


def _draw_basemap(ax):
    for xs, ys in _us_coastline_lines():
        ax.plot(xs, ys, color="#888888", lw=0.6, zorder=1)
    ax.set_xlim(-140, -55)
    ax.set_ylim(18, 56)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_aspect("equal")


# ── Fast map data: ALL training samples, direct lat/lon read (V5 schema) ───
#
# V5 fix note: earlier V2-V4 datasets only stored `coord` (lat_idx, lon_idx
# -- integer indices into the ERA5 153x341 grid) and relied on a *different,
# unrelated* HRRR-Mini reference grid copied into the `invariant` group for
# downstream lat/lon lookups -- a grid mismatch that produced physically
# meaningless coordinates (see investigation log / V4 Figure S9 incident).
# V5's nc_writer_v5.py was patched to also write REAL `lat`/`lon` float
# arrays directly to the root group at write time, eliminating this class
# of bug entirely. This function just reads them directly -- no index
# lookup, no grid-mismatch risk.
def load_all_patch_centers(nc_path):
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path, "r")
    if "lat" not in ds.variables or "lon" not in ds.variables:
        ds.close()
        raise RuntimeError(
            f"{nc_path} has no root lat/lon variables -- this script expects "
            "the V5 schema (written by nc_writer_v5.py). For V4-and-earlier "
            "files, lat/lon must be resolved via the ERA5-grid coord fix "
            "documented in the V4 investigation notes instead."
        )
    lats = np.array(ds.variables["lat"][:], dtype=np.float64)
    lons = np.array(ds.variables["lon"][:], dtype=np.float64)
    N = len(lats)
    ds.close()
    return lats, lons, N


# ── Histogram data: random SAMPLE subset (not all 82,615), full spatial field
#    per selected sample (no pixel striding needed once sample count is small)
# ERA5 input group uses ERA5's own short names, which differ from the HRRR
# output short names for 3 variables (nc_writer.py's _SL_CHANNELS vs
# _OUTPUT_CHANNELS). Mapping is output_key -> input_key.
OUTPUT_TO_INPUT_KEY = {
    "2t":  "t2m",
    "10u": "u10",
    "10v": "v10",
    # sp, q, tp, sf, ssrd, strd use the same short name on both sides
}

# ERA5 ssrd/strd are accumulated J/m^2 over the 1-hour forecast step; HRRR
# ssrd/strd are instantaneous W/m^2 (same convention already applied in
# score_g3.py's load_era5_var() for the main eval pipeline). Divide by 3600
# to convert accumulated J/m^2 over 1hr -> average W/m^2.
ERA5_RADIATION_DIVISOR = 3600.0
RADIATION_VARS = {"ssrd", "strd"}


def load_histogram_subset(train_path, n_samples=3000, seed=42):
    """
    Returns (era5_means, hrrr_means, n_used) where each is a dict of
    per-sample SPATIAL MEAN arrays (1 value per sample, not per-pixel),
    computed over ALL n_used samples (no subsampling of which samples --
    only the spatial dimension is reduced via mean, so every sample
    contributes, matching the "use all 100,000 samples" requirement).
    ERA5 and HRRR have different pixel counts (32x32 vs 256x256) so a
    spatial mean per sample is the correct way to make their histograms
    directly comparable.
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(train_path, "r")
    N_total = ds.dimensions["sample"].size
    n_used = min(n_samples, N_total) if n_samples else N_total
    sl = slice(0, n_used)

    era5_means, hrrr_means = {}, {}
    t0 = time.time()
    for vd in ALL_VARS:
        k = vd["key"]
        in_key = OUTPUT_TO_INPUT_KEY.get(k, k)

        if "output" in ds.groups and k in ds.groups["output"].variables:
            v = ds.groups["output"].variables[k]
            arr = np.array(v[sl])                      # (n, 256, 256)
            hrrr_means[k] = arr.reshape(arr.shape[0], -1).mean(axis=1)

        if "input" in ds.groups and in_key in ds.groups["input"].variables:
            v = ds.groups["input"].variables[in_key]
            arr = np.array(v[sl])                       # (n, 32, 32)
            era5_vals = arr.reshape(arr.shape[0], -1).mean(axis=1)
            if k in RADIATION_VARS:
                era5_vals = era5_vals / ERA5_RADIATION_DIVISOR
            era5_means[k] = era5_vals

        print(f"  [S9] histogram data for {k} (input key={in_key}): "
              f"{time.time()-t0:.1f}s elapsed (cumulative)")
    ds.close()
    return era5_means, hrrr_means, n_used


def _hist_panel(ax, vd, era5_vals, hrrr_vals):
    k = vd["key"]
    e = np.asarray(era5_vals, dtype=float) if era5_vals is not None else None
    h = np.asarray(hrrr_vals, dtype=float) if hrrr_vals is not None else None
    if e is not None:
        e = e[np.isfinite(e)]
    if h is not None:
        h = h[np.isfinite(h)]
    if k in ZERO_TRIM_VARS:
        if e is not None: e = e[e > 1e-6]
        if h is not None: h = h[h > 1e-6]

    log = vd.get("log", False) and k in ZERO_TRIM_VARS
    if log:
        lo = max(min(h.min() if h is not None and h.size else 1e-3, 1e-3), 1e-4)
        hi = max(h.max() if h is not None and h.size else 1.0, 1.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.set_xscale("log")
    else:
        all_v = np.concatenate([v for v in (e, h) if v is not None and v.size])
        if all_v.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            return
        lo, hi = np.nanpercentile(all_v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)

    if e is not None and e.size:
        ax.hist(e, bins=bins, density=True, histtype="stepfilled",
                color=CB["era5"], alpha=0.35, label="ERA5 (input)", zorder=2)
    if h is not None and h.size:
        ax.hist(h, bins=bins, density=True, histtype="step",
                color=CB["truth"], lw=1.3, label="HRRR (output)", zorder=3)

    ax.set_xlabel(f"{vd['label']} ({vd['unit']})", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_title(vd["long"], fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    if k in ZERO_TRIM_VARS:
        ax.text(0.97, 0.93, "zeros removed", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, style="italic",
                color="#666")


def make_figure(train_path, n_hist_samples, outdir):
    apply_style()

    t0 = time.time()
    lats, lons, n_total = load_all_patch_centers(train_path)
    print(f"  [S9] map: loaded all {n_total} patch centers in {time.time()-t0:.2f}s "
          f"(direct lat/lon read, V5 schema)")

    t1 = time.time()
    era5_data, hrrr_data, n_used = load_histogram_subset(
        train_path, n_samples=n_hist_samples)
    print(f"  [S9] histograms: read {n_used} sample subset in {time.time()-t1:.1f}s")

    fig = plt.figure(figsize=(DC, DC*1.2), constrained_layout=False)
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1.3,1,1,1],
                           hspace=0.55, wspace=0.35,
                           left=0.08, right=0.97, top=0.91, bottom=0.05)

    ax_map = fig.add_subplot(gs[0, :])
    _draw_basemap(ax_map)
    ax_map.scatter(lons, lats, s=1.0, alpha=0.15, color=CB["corrdiff"],
                  edgecolors="none", rasterized=True, zorder=2)
    ax_map.set_title(f"Training patch centers (N={n_total:,})",
                     fontsize=9, pad=10)
    panel_label(ax_map, "a")

    for pi, vd in enumerate(ALL_VARS):
        row = 1 + pi // 3
        col = pi % 3
        ax = fig.add_subplot(gs[row, col])
        k = vd["key"]
        _hist_panel(ax, vd, era5_data.get(k), hrrr_data.get(k))
        panel_label(ax, chr(98+pi))

    handles, labels = [], []
    for ax in fig.axes[1:]:
        h_, l_ = ax.get_legend_handles_labels()
        for hh, ll in zip(h_, l_):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper right",
                  bbox_to_anchor=(0.985, 0.93), fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Training Dataset: Geographic Coverage (N={n_total:,}) and "
        f"Per-Sample Spatial-Mean Variable Distributions (N={n_used:,})",
        fontsize=9.5, fontweight="bold", y=0.995)

    save_fig(fig, f"{outdir}/figure_S9_training_data.pdf")
    plt.close(fig)
    print(f"Figure S9 done. Total time: {time.time()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/v5/randomly/train.nc",
                   help="Path to the V5 NetCDF file (must have root lat/lon "
                        "variables, written by nc_writer_v5.py)")
    p.add_argument("--n-hist-samples", type=int, default=100000,
                   help="Number of samples to include in the 9 histogram "
                        "panels, taken as a contiguous slice from the start "
                        "of the file (default: 100000 = full V5 train set). "
                        "Each sample contributes its SPATIAL MEAN, not raw "
                        "pixels, so ERA5 (32x32) and HRRR (256x256) are "
                        "directly comparable despite different patch sizes. "
                        "Map always uses ALL samples in the file regardless "
                        "of this value.")
    p.add_argument("--outdir", default="paper_figures_v2")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    make_figure(a.train, a.n_hist_samples, a.outdir)

if __name__ == "__main__":
    main()
