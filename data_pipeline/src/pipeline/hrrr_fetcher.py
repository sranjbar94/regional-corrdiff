"""
HRRR Fetcher — streams HRRR GRIB2 from Google Cloud Storage and extracts
patches centred on given (lat, lon) coordinates.

Optimized: downloads each GRIB2 file ONCE per timestamp, then extracts
multiple patches from the parsed in-memory data.

Source: Google Cloud Storage (much faster than AWS S3 from Yale Bouchet)
URL pattern: https://storage.googleapis.com/high-resolution-rapid-refresh/
             hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfXX.grib2
"""

from __future__ import annotations
import os
import socket
import tempfile
import time
import warnings
from datetime import datetime

import numpy as np

from src.utils.logger import get_logger

log = get_logger("hrrr_fetcher")

# Map our output channel names -> list of candidate cfgrib variable names
_HRRR_VAR_MAP = {
    "2t":   ["t2m"],
    "10u":  ["u10"],
    "10v":  ["v10"],
    "sp":   ["sp", "pres"],
    "q":    ["sh2", "q2m", "q"],
    "ssrd": ["sdswrf", "dswrf"],
    "strd": ["sdlwrf", "dlwrf"],
    "tp":   ["tp", "prate"],
    "sf":   ["sdwe", "weasd", "sde"],
}

GCS_BASE = "https://storage.googleapis.com/high-resolution-rapid-refresh"


def _gcs_url(dt: datetime, fxx: int = 0) -> str:
    return (
        f"{GCS_BASE}/hrrr.{dt.strftime('%Y%m%d')}/"
        f"conus/hrrr.t{dt.hour:02d}z.wrfsfcf{fxx:02d}.grib2"
    )


def _find_patch_indices(
    lats: np.ndarray, lons: np.ndarray,
    lat: float, lon: float, patch_size: int
) -> tuple[int, int] | None:
    """Find row/col of nearest grid point; return None if too close to edge."""
    lon360 = lon % 360
    dist = np.sqrt((lats - lat) ** 2 + (lons - lon360) ** 2)
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    half = patch_size // 2
    if r - half < 0 or c - half < 0:
        return None
    if r + half > lats.shape[0] or c + half > lats.shape[1]:
        return None
    return int(r), int(c)


# ======================================================================
# ParsedHRRR — one GRIB2 file in memory for multi-patch extraction
# ======================================================================

class ParsedHRRR:
    """
    Holds a parsed HRRR GRIB2 file in memory for efficient multi-patch
    extraction.

    Download ~120 MB GRIB2 from GCS:  ~2-4 seconds
    Parse with cfgrib:                ~1-2 seconds
    Extract one 64x64 patch:          <1 millisecond
    Memory per file:                  ~80 MB (numpy arrays)
    """

    def __init__(self, all_vars: dict):
        self._all_vars = all_vars

    def extract_patch(
        self, lat: float, lon: float, patch_size: int = 64
    ) -> dict[str, np.ndarray] | None:
        patch = {}
        rc_cache = {}

        for out_name, candidates in _HRRR_VAR_MAP.items():
            for cand in candidates:
                if cand.lower() in self._all_vars:
                    data, lats, lons = self._all_vars[cand.lower()]

                    grid_key = (lats.shape, lons.shape)
                    if grid_key not in rc_cache:
                        rc_cache[grid_key] = _find_patch_indices(
                            lats, lons, lat, lon, patch_size
                        )
                    rc = rc_cache[grid_key]

                    if rc is None:
                        return None

                    r, c = rc
                    half = patch_size // 2
                    arr = data[r - half: r + half, c - half: c + half]
                    if arr.shape == (patch_size, patch_size):
                        patch[out_name] = arr.astype(np.float32)
                    break

        return patch if patch else None

    @property
    def n_vars(self) -> int:
        return len(self._all_vars)


# ======================================================================
# fetch_hrrr_file — download once, parse once (WITH TIMEOUT FIX)
# ======================================================================

def fetch_hrrr_file(
    dt: datetime,
    fxx: int = 0,
    retries: int = 3,
    dry_run: bool = False,
) -> ParsedHRRR | None:
    """
    Download and parse a full HRRR GRIB2 file for the given datetime.
    Returns a ParsedHRRR object for multi-patch extraction, or None on failure.

    Uses Google Cloud Storage (GCS) instead of AWS S3 for much faster
    download speeds from Yale Bouchet HPC.
    
    FIX: Added socket-level timeout to prevent thread pool deadlock.
    """
    if dry_run:
        log.debug(f"  [DRY_RUN] synthetic HRRR file for {dt}")
        synth = {}
        nrows, ncols = 1059, 1799
        for out_name, candidates in _HRRR_VAR_MAP.items():
            cand = candidates[0]
            fake_data = np.random.randn(nrows, ncols).astype(np.float32)
            fake_lats = (np.linspace(21, 53, nrows)[:, None]
                         * np.ones((1, ncols))).astype(np.float32)
            fake_lons = (np.linspace(225, 300, ncols)[None, :]
                         * np.ones((nrows, 1))).astype(np.float32)
            synth[cand.lower()] = (fake_data, fake_lats, fake_lons)
        return ParsedHRRR(synth)

    try:
        import cfgrib
        import xarray as xr
        import urllib.request
    except ImportError:
        raise ImportError(
            "cfgrib is required for HRRR fetching.\n"
            "Run: pip install cfgrib eccodes"
        )

    url = _gcs_url(dt, fxx)

    for attempt in range(1, retries + 1):
        tmp_path = None
        try:
            log.debug(f"  [HRRR] fetching {url}  (attempt {attempt}/{retries})")
            t0 = time.time()

            # FIX: Set socket timeout at system level BEFORE urlopen
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)  # 30 second timeout
            
            try:
                req = urllib.request.Request(url)
                # Also set timeout on urlopen itself
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
            finally:
                # Restore original timeout
                socket.setdefaulttimeout(old_timeout)

            dl_time = time.time() - t0
            size_mb = len(data) / 1e6
            log.debug(f"  [HRRR] downloaded {size_mb:.0f} MB in {dl_time:.1f}s")

            # Write to temp file for cfgrib parsing
            with tempfile.NamedTemporaryFile(
                suffix=".grib2", delete=False
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            del data  # free memory

            # Open all message groups in the GRIB2 file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                datasets = cfgrib.open_datasets(tmp_path)

            # Build a flat lookup: cfgrib_varname -> (data, lats, lons)
            all_vars = {}
            for ds in datasets:
                xds = xr.Dataset(ds)
                for vname in xds.data_vars:
                    shape = xds[vname].shape
                    if len(shape) == 2:
                        all_vars[vname.lower()] = (
                            xds[vname].values.copy(),
                            xds["latitude"].values.copy(),
                            xds["longitude"].values.copy(),
                        )
                xds.close()

            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass

            log.debug(f"  [HRRR] parsed {len(all_vars)} 2D vars from {url}")
            return ParsedHRRR(all_vars)

        except socket.timeout:
            log.warning(f"  [HRRR] socket timeout on attempt {attempt}/{retries}")
            time.sleep(2 ** attempt)
        except urllib.error.URLError as e:
            log.warning(f"  [HRRR] URL error on attempt {attempt}/{retries}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"  [HRRR] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    log.warning(f"  [HRRR] all {retries} attempts failed for {dt}")
    return None


# ======================================================================
# Backward-compatible single-patch function
# ======================================================================

def fetch_hrrr_patch(
    dt: datetime,
    lat: float,
    lon: float,
    patch_size: int = 64,
    fxx: int = 0,
    retries: int = 3,
    dry_run: bool = False,
) -> dict[str, np.ndarray] | None:
    parsed = fetch_hrrr_file(dt, fxx=fxx, retries=retries, dry_run=dry_run)
    if parsed is None:
        return None
    return parsed.extract_patch(lat, lon, patch_size)
