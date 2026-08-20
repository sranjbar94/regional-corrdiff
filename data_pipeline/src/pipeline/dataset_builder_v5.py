"""
DatasetBuilder — the main sampling loop (Stage 2).

Supports optional shard mode for parallel builds:
  --shard_id 0 --n_shards 8
  Each shard gets a disjoint set of timestamps from a pre-partitioned file.
  Output and checkpoints are per-shard to avoid conflicts.
"""

from __future__ import annotations
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import zoom

from src.pipeline.era5_reader import ERA5Reader
from src.pipeline.hrrr_fetcher import fetch_hrrr_file
from src.pipeline.nc_writer_v5 import NCWriter
from src.utils.logger import get_logger
from src.utils.time_sampler import TimestampSampler

log = get_logger("dataset_builder")


class DatasetBuilder:
    """
    Orchestrates the full dataset build with parallel HRRR downloads.

    Parameters
    ----------
    cfg           : loaded pipeline config namespace
    dry_run       : if True, use synthetic HRRR data (no S3 calls)
    build_workers : number of parallel HRRR download threads (default: 20)
    shard_id      : if set, use disjoint timestamp partition (0-based)
    n_shards      : total number of shards
    """

    def __init__(self, cfg: SimpleNamespace, dry_run: bool = False,
                 build_workers: int = 20,
                 shard_id: int | None = None,
                 n_shards: int | None = None):
        self.cfg           = cfg
        self.dry_run       = dry_run
        self.build_workers = build_workers
        self.shard_id      = shard_id
        self.n_shards      = n_shards

        self.era5_dir  = Path(cfg.storage.era5_raw_dir)

        # Per-shard output and checkpoint paths
        if shard_id is not None:
            base_name = Path(cfg.storage.output_filename).stem
            suffix = Path(cfg.storage.output_filename).suffix
            out_name = f"{base_name}_shard{shard_id}{suffix}"
            ckpt_name = f"build_checkpoint_shard_{shard_id}.json"
        else:
            out_name = cfg.storage.output_filename
            ckpt_name = "build_checkpoint.json"

        self.out_path  = Path(cfg.storage.output_dir) / out_name
        self.ckpt_dir  = Path(cfg.storage.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True) if not self.ckpt_dir.exists() else None
        self.ckpt_path = self.ckpt_dir / ckpt_name

        self.n_samples  = cfg.sampling.n_samples
        self.per_ts     = cfg.sampling.samples_per_ts
        self.seed       = cfg.sampling.random_seed
        self.ckpt_freq  = cfg.storage.checkpoint_freq
        self.log_freq   = cfg.logging.log_freq

        # Per-shard seed to get different ocean center draws
        if shard_id is not None:
            self.seed = self.seed + shard_id

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Load timestamp partition if in shard mode
        self._allowed_timestamps = None
        if shard_id is not None:
            partition_path = self.ckpt_dir / "timestamp_partitions.json"
            if not partition_path.exists():
                raise FileNotFoundError(
                    f"Timestamp partition file not found: {partition_path}\n"
                    "Run: python scripts/partition_timestamps.py --n-shards N"
                )
            with open(partition_path) as f:
                partitions = json.load(f)
            key = str(shard_id)
            if key not in partitions:
                raise KeyError(
                    f"Shard {shard_id} not found in partition file. "
                    f"Available: {list(partitions.keys())}"
                )
            self._allowed_timestamps = partitions[key]
            log.info(f"Shard mode: shard_id={shard_id}, n_shards={n_shards}, "
                     f"timestamps={len(self._allowed_timestamps):,}")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, samples_done: int, used_ts: set):
        state = {
            "samples_done":    samples_done,
            "used_timestamps": sorted(used_ts),
        }
        if self.shard_id is not None:
            state["shard_id"] = self.shard_id
            state["n_shards"] = self.n_shards
        with open(self.ckpt_path, "w") as f:
            json.dump(state, f, indent=2)
        log.info(f"  [CKPT] checkpoint saved at {samples_done:,} samples")

    def _load_checkpoint(self) -> tuple[int, set]:
        if not self.ckpt_path.exists():
            return 0, set()
        with open(self.ckpt_path) as f:
            state = json.load(f)
        samples_done = state.get("samples_done", 0)
        used_ts      = set(state.get("used_timestamps", []))

        # Safety: verify shard_id matches
        if self.shard_id is not None:
            ckpt_shard = state.get("shard_id")
            if ckpt_shard is not None and ckpt_shard != self.shard_id:
                log.warning(
                    f"Checkpoint shard_id={ckpt_shard} != current {self.shard_id}. "
                    "Starting fresh."
                )
                return 0, set()

        log.info(f"Resuming from checkpoint: {samples_done:,} samples done")
        return samples_done, used_ts

    # ------------------------------------------------------------------
    # Synthetic ERA5 helper (for dry-run / testing)
    # ------------------------------------------------------------------

    def _synthetic_era5_patch(self, lat: float, lon: float) -> dict:
        """Generate a synthetic ERA5 patch for dry-run testing."""
        ps = self.cfg.patches.era5_patch_size
        res = self.cfg.patches.era5_resolution
        patch = {}
        synth_ranges = {
            "t2m": (270, 310), "d2m": (260, 300), "sp": (95000, 105000),
            "msl": (98000, 104000), "u10": (-15, 15), "v10": (-15, 15),
            "ssrd": (0, 3e6), "strd": (0, 2e6), "tp": (0, 0.05),
            "sf": (0, 0.01), "lsm": (0, 0.1), "tcwv": (5, 60),
            "q": (0.001, 0.02),
        }
        for var, (lo, hi) in synth_ranges.items():
            patch[var] = np.random.uniform(lo, hi, (ps, ps)).astype(np.float32)
        for var in ["u", "v", "z", "t", "q"]:
            for lev in [1000, 850, 500, 250]:
                patch[f"{var}{lev}"] = np.random.uniform(
                    -50, 50, (ps, ps)
                ).astype(np.float32)
        half = ps // 2
        patch["_lat"] = np.linspace(
            lat - half * res, lat + half * res, ps
        ).astype(np.float32)
        patch["_lon"] = np.linspace(
            lon - half * res, lon + half * res, ps
        ).astype(np.float32)
        return patch

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.cfg

        shard_label = ""
        if self.shard_id is not None:
            shard_label = f" [SHARD {self.shard_id}/{self.n_shards}]"

        log.info("=" * 60)
        log.info(f"CorrDiff Regional Dataset Builder{shard_label}")
        log.info(f"  Output         : {self.out_path}")
        log.info(f"  Target samples : {self.n_samples:,}")
        log.info(f"  Samples/ts     : {self.per_ts}")
        log.info(f"  Build workers  : {self.build_workers}")
        log.info(f"  Date range     : {cfg.time.date_start} -> {cfg.time.date_end}")
        log.info(f"  Dry run        : {self.dry_run}")
        if self.shard_id is not None:
            log.info(f"  Shard          : {self.shard_id} of {self.n_shards}")
            log.info(f"  Shard ts pool  : {len(self._allowed_timestamps):,}")
        log.info("=" * 60)

        # --- Checkpoint / resume ---
        samples_done, used_ts = self._load_checkpoint()
        resume = samples_done > 0

        # --- ERA5 reader (used sequentially in main thread) ---
        reader = ERA5Reader(self.era5_dir, cfg)

        if self.dry_run:
            log.info("Dry run: generating synthetic ocean patch centres")
            lon_min = cfg.domain.lon_min
            lon_max = cfg.domain.lon_max
            lat_min = cfg.domain.lat_min
            lat_max = cfg.domain.lat_max
            buffer = ((cfg.patches.era5_patch_size // 2)
                      * cfg.patches.era5_resolution)
            ocean_centers = [
                (round(lat, 2), round(lon, 2))
                for lat in np.arange(lat_min + buffer, lat_max - buffer, 1.0)
                for lon in np.arange(lon_min + buffer, lon_max - buffer, 1.0)
            ]
            random.shuffle(ocean_centers)
            ocean_centers = ocean_centers[:5000]
            log.info(f"Synthetic ocean patch centres: {len(ocean_centers):,}")
        else:
            ocean_centers = reader.get_ocean_patch_centers(
                lsm_threshold  = cfg.ocean.lsm_threshold,
                min_ocean_frac = cfg.ocean.min_ocean_frac,
                lon_min = cfg.domain.lon_min,
                lon_max = cfg.domain.lon_max,
                lat_min = cfg.domain.lat_min,
                lat_max = cfg.domain.lat_max,
            )
        if not ocean_centers:
            raise RuntimeError(
                "No valid ocean patch centres found. "
                "Check ERA5 files and domain bounds in config."
            )

        # --- Timestamp sampler ---
        sampler = TimestampSampler(
            date_start    = cfg.time.date_start,
            date_end      = cfg.time.date_end,
            allow_repeats = cfg.sampling.allow_ts_repeats,
            seed          = self.seed,
            used          = used_ts,
            allowed_timestamps = self._allowed_timestamps,
        )
        log.info(f"Timestamp pool: {sampler.pool_size:,} "
                 f"({sampler.n_used:,} already used)")

        # --- NetCDF writer (used sequentially in main thread) ---
        writer = NCWriter(self.out_path, self.n_samples, cfg)
        writer.open(resume=resume)

        # Get LSM grid for coord index lookup
        if self.dry_run:
            res = cfg.patches.era5_resolution
            lsm_lats = np.arange(
                cfg.domain.lat_max, cfg.domain.lat_min, -res
            ).astype(np.float32)
            lsm_lons = np.arange(
                cfg.domain.lon_min, cfg.domain.lon_max, res
            ).astype(np.float32)
        else:
            _, lsm_lats, lsm_lons = reader.get_lsm()

        # --- Sampling loop with parallel HRRR downloads ---
        log.info(f"Sampling loop: {samples_done:,} -> {self.n_samples:,}")
        batch_size = self.build_workers  # timestamps per batch

        try:
            while samples_done < self.n_samples:
                # ======================================================
                # STEP 1: Draw a batch of timestamps
                # ======================================================
                batch_timestamps = []
                for _ in range(batch_size):
                    try:
                        dt = sampler.sample()
                        batch_timestamps.append(dt)
                    except StopIteration as e:
                        log.warning(str(e))
                        break

                if not batch_timestamps:
                    log.warning("No more timestamps available. Stopping.")
                    break

                # ======================================================
                # STEP 2: Download HRRR files in PARALLEL with timeout
                # ======================================================
                log.info(
                    f"  [BATCH] Downloading {len(batch_timestamps)} HRRR "
                    f"files in parallel ({self.build_workers} workers)..."
                )

                hrrr_results = {}
                with ThreadPoolExecutor(
                    max_workers=self.build_workers
                ) as pool:
                    futures = {
                        pool.submit(
                            fetch_hrrr_file,
                            dt,
                            fxx=cfg.hrrr.forecast_hour,
                            dry_run=self.dry_run,
                        ): dt
                        for dt in batch_timestamps
                    }
                    # FIX: Add timeout to as_completed to prevent infinite hang
                    for future in as_completed(futures, timeout=60):
                        dt = futures[future]
                        try:
                            hrrr_results[dt] = future.result()
                        except Exception as e:
                            log.warning(
                                f"  [HRRR] Failed for {dt}: {e}"
                            )
                            hrrr_results[dt] = None

                ok = sum(1 for v in hrrr_results.values() if v is not None)
                log.info(
                    f"  [BATCH] HRRR downloads: {ok}/"
                    f"{len(batch_timestamps)} succeeded"
                )

                # ======================================================
                # STEP 3: Process each timestamp SEQUENTIALLY
                # ======================================================
                for dt in batch_timestamps:
                    if samples_done >= self.n_samples:
                        break

                    parsed_hrrr = hrrr_results.get(dt)
                    if parsed_hrrr is None:
                        log.debug(
                            f"  Skipping timestamp {dt} (HRRR failed)"
                        )
                        continue

                    # Pick random ocean centres for this timestamp
                    centres_this_ts = random.sample(
                        ocean_centers,
                        min(self.per_ts, len(ocean_centers)),
                    )

                    ts_successes = 0

                    for lat, lon in centres_this_ts:
                        if samples_done >= self.n_samples:
                            break

                        try:
                            # ERA5 patch (local disk — fast)
                            if self.dry_run:
                                era5_patch = self._synthetic_era5_patch(
                                    lat, lon
                                )
                            else:
                                era5_patch = reader.extract_patch(
                                    dt, lat, lon
                                )

                            # HRRR patch (in memory — instant)
                            hrrr_patch = parsed_hrrr.extract_patch(
                                lat, lon,
                                patch_size=cfg.patches.hrrr_patch_size,
                            )
                            if hrrr_patch is None:
                                log.debug(
                                    f"  HRRR patch failed: {dt} "
                                    f"({lat:.2f}, {lon:.2f})"
                                )
                                continue

                            # Fill missing strd from ERA5
                            if ("strd" not in hrrr_patch
                                    and "strd" in era5_patch):
                                hr = cfg.patches.hrrr_patch_size
                                lr = cfg.patches.era5_patch_size
                                scale = hr / lr
                                strd_lr = era5_patch["strd"]
                                hrrr_patch["strd"] = zoom(
                                    strd_lr, scale, order=1
                                ).astype(np.float32)

                            # ERA5 coord indices
                            lat_idx = int(
                                np.argmin(np.abs(lsm_lats - lat))
                            )
                            lon_idx = int(
                                np.argmin(np.abs(lsm_lons - lon))
                            )

                            # Write to NetCDF (sequential)
                            writer.write_sample(
                                idx        = samples_done,
                                dt         = dt,
                                lat_idx    = lat_idx,
                                lon_idx    = lon_idx,
                                lat        = lat,
                                lon        = lon,
                                era5_patch = era5_patch,
                                hrrr_patch = hrrr_patch,
                            )

                            samples_done += 1
                            ts_successes += 1

                            # Logging
                            if samples_done % self.log_freq == 0:
                                pct = (100 * samples_done
                                       / self.n_samples)
                                log.info(
                                    f"  Progress: {samples_done:>8,} / "
                                    f"{self.n_samples:,}  ({pct:.1f}%)"
                                )
                            # Checkpoint
                            if samples_done % self.ckpt_freq == 0:
                                writer.flush()
                                self._save_checkpoint(
                                    samples_done, sampler.used
                                )
                        except Exception as e:
                            import traceback as _tb
                            log.warning(
                                f"  Sample failed "
                                f"[{dt} ({lat:.2f},{lon:.2f})]: {e}\n"
                                f"{''.join(_tb.format_exc())}"
                            )
                            continue
                    if ts_successes == 0:
                        log.debug(
                            f"  Zero successes from timestamp {dt}"
                        )
                    # Free HRRR memory after processing this timestamp
                    hrrr_results[dt] = None
        finally:
            writer.close()
            self._save_checkpoint(samples_done, sampler.used)
        log.info("=" * 60)
        log.info(
            f"Build complete{shard_label}: {samples_done:,} / "
            f"{self.n_samples:,} samples"
        )
        log.info(f"Output: {self.out_path}")
        log.info("=" * 60)
        return samples_done
