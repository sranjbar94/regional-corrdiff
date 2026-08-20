"""
_rapsd_core.py — Correct radially-averaged power spectral density.

Fixes the "sudden drop" artifact via:
  1. Planar detrend (remove mean + linear ramp) to reduce edge discontinuity
  2. 2-D Hann window (removes spectral leakage / Gibbs cliff)
  3. Proper window power normalization
  4. Radial averaging with log-spaced wavenumber bins
  5. Clip to the reliable wavenumber range (drop the lowest-2 and Nyquist bins)
"""
import numpy as np


def _planar_detrend(field):
    """Subtract least-squares plane to reduce edge wrap discontinuity."""
    H, W = field.shape
    y, x = np.mgrid[0:H, 0:W]
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(H*W)])
    coef, *_ = np.linalg.lstsq(A, field.ravel(), rcond=None)
    plane = (A @ coef).reshape(H, W)
    return field - plane


def _hann2d(H, W):
    wy = np.hanning(H)
    wx = np.hanning(W)
    return np.outer(wy, wx)


def rapsd(field, pixel_km=3.0, n_bins=40):
    """
    Radially-averaged PSD of a 2-D field.

    Returns (wavelength_km, psd) sorted by ascending wavenumber
    (i.e. descending wavelength). Power-law range is clean; the
    unreliable lowest wavenumbers and Nyquist bin are trimmed.
    """
    field = np.asarray(field, dtype=float)
    H, W = field.shape

    # 1. detrend + window
    f = _planar_detrend(field)
    win = _hann2d(H, W)
    f = f * win
    win_power = (win**2).mean()   # normalization for windowing

    # 2. 2-D FFT power
    F = np.fft.fftshift(np.fft.fft2(f))
    psd2d = (np.abs(F)**2) / (H * W * win_power)

    # 3. radial wavenumber grid (cycles per pixel)
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    kr = np.sqrt((x - cx)**2 + (y - cy)**2)   # radius in pixels (freq domain)
    kr_flat = kr.ravel()
    psd_flat = psd2d.ravel()

    # 4. radial average in linear wavenumber bins
    k_max = min(cx, cy)
    bins = np.arange(1, k_max + 1)
    psd_radial = np.zeros(len(bins) - 1)
    k_centers = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        m = (kr_flat >= bins[i]) & (kr_flat < bins[i+1])
        if m.any():
            psd_radial[i] = psd_flat[m].mean()
        k_centers[i] = 0.5 * (bins[i] + bins[i+1])

    # 5. convert radial index → wavenumber (cycles/pixel) → wavelength (km)
    wavenumber = k_centers / H              # cycles per pixel
    wavelength = pixel_km / wavenumber      # km

    # 6. Keep a FIXED set of bins (drop only the first largest-scale bin and
    #    the Nyquist bin) so every call returns the same length. Zero/negative
    #    PSD bins are set to NaN rather than removed, preserving array shape
    #    (critical for averaging across precipitation patches with empty rings).
    ps = psd_radial.copy()
    ps[ps <= 0] = np.nan
    wl = wavelength[1:-1]
    ps = ps[1:-1]

    order = np.argsort(wl)
    return wl[order], ps[order]


def mean_rapsd(nc_path, var_key, group, n_samples=200, seed=42,
               wind=False, blur_sigma=None, pixel_km=3.0):
    """Average RAPSD over n_samples patches. Uses nanmean to tolerate
    empty wavenumber bins (common for sparse precipitation fields)."""
    import netCDF4 as nc4
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    ds = nc4.Dataset(nc_path, "r")
    first_var = list(ds.groups[group].variables.keys())[0]
    N = ds.groups[group].variables[first_var].shape[0]
    idx = rng.choice(N, min(n_samples, N), replace=False)

    all_psd = []
    wl_ref = None
    for i in idx:
        if wind:
            u = np.array(ds.groups[group].variables["10u"][i])
            v = np.array(ds.groups[group].variables["10v"][i])
            field = np.sqrt(u**2 + v**2)
        else:
            field = np.array(ds.groups[group].variables[var_key][i])
        if blur_sigma:
            field = gaussian_filter(field, sigma=blur_sigma)
        wl, ps = rapsd(field, pixel_km=pixel_km)
        all_psd.append(ps)
        wl_ref = wl
    ds.close()
    return wl_ref, np.nanmean(np.array(all_psd), axis=0)
