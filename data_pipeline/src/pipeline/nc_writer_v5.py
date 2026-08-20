"""
NetCDF Writer — creates and writes the regional_hrrr_train_dataset.nc
in a schema that exactly matches HRRR-Mini so it drops into CorrDiff
without any dataset loader changes.

HRRR-Mini schema recap:
    Dimensions : sample, y_lr, x_lr, y_hr, x_hr, coord
    Root vars  : time(sample), coord(sample, coord)
    Group input   : (sample, y_lr, x_lr)  -- ERA5 LR channels
    Group output  : (sample, y_hr, x_hr)  -- HRRR HR targets
    Group invariant: (y_lr, x_lr)          -- static fields (written once)
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np

from src.utils.logger import get_logger

log = get_logger("nc_writer")

# Epoch aligned with the dataset period (2015-01-01)
_EPOCH_UNIX = 0  # Unix epoch: 1970-01-01


# All ERA5 input channel names written to [input] group
_SL_CHANNELS = [
    "u10", "v10", "t2m", "tcwv", "sp", "msl",
    "d2m", "q",
    "ssrd", "strd", "tp", "sf", "lsm",
]
_PL_BASE = ["u", "v", "z", "t"]
_PL_LEVS = [1000, 850, 500, 250]


def _all_input_channels() -> list[str]:
    channels = list(_SL_CHANNELS)
    for var in _PL_BASE:
        for lev in _PL_LEVS:
            channels.append(f"{var}{lev}")
    # specific humidity at pressure levels
    for lev in _PL_LEVS:
        channels.append(f"q{lev}")
    return channels


# HRRR output short names (must match config hrrr.output_vars[*].short)
_OUTPUT_CHANNELS = ["2t", "10u", "10v", "tp", "ssrd", "strd", "sp", "q", "sf"]


class NCWriter:
    """
    Manages creation of and incremental writing to the output NetCDF.
    Call `open()` before the sampling loop, `write_sample()` per sample,
    `close()` when done.  Supports append mode for checkpoint recovery.
    """

    def __init__(self, path: Path | str, n_samples: int, cfg: SimpleNamespace):
        self.path      = Path(path)
        self.n_samples = n_samples
        self.patch_lr  = cfg.patches.era5_patch_size   # 8
        self.patch_hr  = cfg.patches.hrrr_patch_size   # 64
        self.compress  = cfg.storage.compression_level  # 4
        self._ds       = None
        self._invariant_written = False

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    def _create(self):
        """Create a fresh NetCDF file with the HRRR-Mini schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ds = nc.Dataset(str(self.path), "w", format="NETCDF4")

        # --- Dimensions ---
        ds.createDimension("sample", None)  # unlimited for efficient appending
        ds.createDimension("y_lr",   self.patch_lr)
        ds.createDimension("x_lr",   self.patch_lr)
        ds.createDimension("y_hr",   self.patch_hr)
        ds.createDimension("x_hr",   self.patch_hr)
        ds.createDimension("y_grid", 1059)
        ds.createDimension("x_grid", 1799)
        ds.createDimension("coord",  2)

        # --- Root variables ---
        tv = ds.createVariable("time", "i8", ("sample",), chunksizes=(1000,))
        tv.units     = "seconds since 1970-01-01 00:00:00"
        tv.calendar  = "standard"
        tv.long_name = "UTC timestamp of sample"

        cv = ds.createVariable("coord", "u2", ("sample", "coord"), chunksizes=(1000, 2))
        cv.long_name = "ERA5 grid indices of patch centre (lat_idx, lon_idx)"

        latv = ds.createVariable("lat", "f4", ("sample",), chunksizes=(1000,))
        latv.long_name = "Patch centre latitude (degrees N) -- ERA5 grid value, real coordinate"
        lonv = ds.createVariable("lon", "f4", ("sample",), chunksizes=(1000,))
        lonv.long_name = "Patch centre longitude (degrees E, -180 to 180) -- ERA5 grid value, real coordinate"

        # --- [input] group -- ERA5 LR ---
        grp_in = ds.createGroup("input")
        for ch in _all_input_channels():
            v = grp_in.createVariable(
                ch, "f4", ("sample", "y_lr", "x_lr"),
                zlib=True, complevel=self.compress,
                chunksizes=(1, self.patch_lr, self.patch_lr),
                fill_value=np.float32(np.nan),
            )
            v.long_name = ch

        # --- [output] group -- HRRR HR ---
        grp_out = ds.createGroup("output")
        for ch in _OUTPUT_CHANNELS:
            v = grp_out.createVariable(
                ch, "f4", ("sample", "y_hr", "x_hr"),
                zlib=True, complevel=self.compress,
                chunksizes=(1, self.patch_hr, self.patch_hr),
                fill_value=np.float32(np.nan),
            )
            v.long_name = ch

        # --- [invariant] group -- static fields ---
        grp_inv = ds.createGroup("invariant")
        for field in ["latitude", "longitude", "elev_mean", "lsm_mean"]:
            v = grp_inv.createVariable(
                field, "f4", ("y_grid", "x_grid"),
                zlib=True, complevel=self.compress,
            )
            v.long_name = field

        # --- Global attributes ---
        ds.title         = "CorrDiff Regional HRRR Training Dataset"
        ds.source_era5   = "ERA5 reanalysis (Copernicus CDS), 0.25 deg"
        ds.source_hrrr   = "HRRR operational (AWS Open Data), ~3 km"
        ds.era5_patch    = f"{self.patch_lr}x{self.patch_lr} pixels"
        ds.hrrr_patch    = f"{self.patch_hr}x{self.patch_hr} pixels"
        ds.schema_compat = "HRRR-Mini (PhysicsNeMo CorrDiff)"
        ds.time_period   = "2015-2025"
        ds.created_utc   = datetime.utcnow().isoformat()

        ds.close()
        log.info(f"Created output NetCDF: {self.path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self, resume: bool = False):
        """Open the NetCDF for writing.  Creates file if not resuming."""
        if not self.path.exists() or not resume:
            self._create()
            self._invariant_written = False
        else:
            log.info(f"Resuming -- appending to: {self.path}")
            self._invariant_written = True   # assume written in prior run
        self._ds = nc.Dataset(str(self.path), "a")

    def write_sample(
        self,
        idx: int,
        dt: datetime,
        lat_idx: int,
        lon_idx: int,
        lat: float,
        lon: float,
        era5_patch: dict,
        hrrr_patch: dict,
    ):
        """
        Write one sample at position `idx`.

        Parameters
        ----------
        idx        : sample index (0-based)
        dt         : sample datetime
        lat_idx    : ERA5 grid lat index of patch centre (kept for backward
                     compatibility; do NOT combine with the invariant group's
                     lat/lon grid -- that grid is a different, unrelated HRRR
                     reference grid, not the ERA5 grid these indices are into)
        lon_idx    : ERA5 grid lon index of patch centre (see above)
        lat        : REAL patch centre latitude (degrees N) -- use this for
                     any plotting/mapping, not lat_idx + invariant grid
        lon        : REAL patch centre longitude (degrees E, -180 to 180)
        era5_patch : dict from ERA5Reader.extract_patch()
        hrrr_patch : dict from fetch_hrrr_patch() -- keys are short names
        """
        import calendar as _cal; self._ds["time"][idx] = int(_cal.timegm(dt.timetuple()))
        self._ds["coord"][idx, :] = [lat_idx, lon_idx]
        self._ds["lat"][idx] = lat
        self._ds["lon"][idx] = lon

        grp_in  = self._ds["input"]
        grp_out = self._ds["output"]
        grp_inv = self._ds["invariant"]

        # Input channels
        for ch in _all_input_channels():
            if ch in era5_patch:
                grp_in[ch][idx] = era5_patch[ch]

        # Output channels
        for ch in _OUTPUT_CHANNELS:
            if ch in hrrr_patch:
                grp_out[ch][idx] = hrrr_patch[ch]

        # Invariants -- written once from HRRR-Mini reference grid (1059x1799)
        if not self._invariant_written:
            import netCDF4 as _nc4
            _HRRR_MINI = (
                "/nfs/roberts/pi/pi_ey239/sr2723/corrdiff_pipeline/"
                "data/hrrr_MINI_totest/hrrr_mini_train.nc"
            )
            _ds_ref = _nc4.Dataset(_HRRR_MINI, "r")
            _inv = _ds_ref.groups["invariant"]
            grp_inv["latitude"][:]  = _inv.variables["latitude"][:]
            grp_inv["longitude"][:] = _inv.variables["longitude"][:]
            grp_inv["elev_mean"][:] = _inv.variables["elev_mean"][:]
            grp_inv["lsm_mean"][:]  = _inv.variables["lsm_mean"][:]
            _ds_ref.close()
            log.info("Invariant written from HRRR-Mini reference grid (1059x1799)")
            self._invariant_written = True

    def flush(self):
        """Force write buffers to disk."""
        if self._ds:
            self._ds.sync()

    def close(self):
        if self._ds:
            self._ds.close()
            self._ds = None
            log.info(f"NetCDF closed: {self.path}")
