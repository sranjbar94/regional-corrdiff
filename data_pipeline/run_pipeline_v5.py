#!/usr/bin/env python3
"""
run_pipeline.py — single entry point for all pipeline stages.

V2: Supports train/test split (2015-2023 → train, 2024-2025 → test)

Usage
-----
# Test run (10 days in Jan 2015, fast validation)
python run_pipeline.py download --test_mode --workers 2
python run_pipeline.py build   --test_mode --dry_run   # synthetic HRRR
python run_pipeline.py build   --test_mode             # real HRRR S3

# Full 10-year run (2015-2025)
python run_pipeline.py download [--workers 2] [--skip_pressure]
python run_pipeline.py build    --split train          # 2015-2023, 500k samples
python run_pipeline.py build    --split test           # 2024-2025, 50k samples
python run_pipeline.py stats    --split train
python run_pipeline.py stats    --split test

# Sharded build (parallel jobs, disjoint timestamps)
python run_pipeline.py build --split train --shard_id 0 --n_shards 8 --n_samples 62500
python run_pipeline.py build --split train --shard_id 1 --n_shards 8 --n_samples 62500
# ... etc

# Custom date range override
python run_pipeline.py download --date_start 2020-01-01 --date_end 2020-01-31
python run_pipeline.py build --split train --n_samples 500 --samples_per_ts 5
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("run_pipeline", log_dir="logs")

# ---------------------------------------------------------------------------
# Test-mode constants: 10 days in Jan 2015
# ---------------------------------------------------------------------------
TEST_DATE_START = "2015-01-01"
TEST_DATE_END   = "2015-01-10"
TEST_N_SAMPLES  = 200    # ~20 samples/day x 10 days

# ---------------------------------------------------------------------------
# Train/Test split defaults
# ---------------------------------------------------------------------------
SPLIT_DATES = {
    "train": ("2015-01-01", "2023-12-31", 500000),   # 500k samples
    "test":  ("2024-01-01", "2025-12-31", 50000),    # 50k samples
}


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------

def cmd_download(cfg, args):
    from src.pipeline.era5_downloader import run_downloader
    log.info("=== Stage 1: ERA5 Download ===")

    start = args.date_start
    end   = args.date_end

    if args.test_mode:
        log.info(
            f"[TEST MODE] Downloading only {TEST_DATE_START} -> {TEST_DATE_END} "
            "(10 days). Re-run without --test_mode for the full 2015-2025 period."
        )
        start = start or TEST_DATE_START
        end   = end   or TEST_DATE_END

    run_downloader(
        cfg,
        workers=args.workers,
        skip_pressure=args.skip_pressure,
        date_start_override=start,
        date_end_override=end,
    )


def cmd_build(cfg, args):
    from src.pipeline.dataset_builder_v5 import DatasetBuilder
    log.info("=== Stage 2: Dataset Build ===")

    # Determine date range and sample count
    if args.test_mode:
        # Test mode: always use 10-day window, small sample count
        date_start = TEST_DATE_START
        date_end = TEST_DATE_END
        n_samples = TEST_N_SAMPLES
        log.info(
            f"[TEST MODE] Building {n_samples} samples from "
            f"{date_start} -> {date_end}. "
            "Re-run without --test_mode for the full dataset."
        )
    elif args.split:
        # Split mode: use predefined train/test ranges
        split = args.split
        if split not in SPLIT_DATES:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'test'.")
        date_start, date_end, n_samples = SPLIT_DATES[split]
        log.info(f"[SPLIT MODE] Building {split} set: {date_start} -> {date_end}, {n_samples} samples")
    else:
        # Custom mode: use provided dates or config defaults
        date_start = args.date_start or cfg.time.date_start
        date_end = args.date_end or cfg.time.date_end
        n_samples = args.n_samples or cfg.sampling.n_samples
        log.info(f"[CUSTOM MODE] Building: {date_start} -> {date_end}, {n_samples} samples")

    # Apply explicit CLI overrides (highest priority)
    if args.n_samples is not None:
        n_samples = args.n_samples

    # Set config values
    cfg.time.date_start = date_start
    cfg.time.date_end = date_end
    cfg.sampling.n_samples = n_samples

    if args.samples_per_ts is not None:
        cfg.sampling.samples_per_ts = args.samples_per_ts

    # Set allow_ts_repeats for small windows (test mode)
    if args.test_mode:
        cfg.sampling.allow_ts_repeats = True

    # Override output filename for splits (non-test mode)
    if args.split and not args.test_mode:
        base_name = Path(cfg.storage.output_filename).stem
        suffix = Path(cfg.storage.output_filename).suffix
        cfg.storage.output_filename = f"{base_name}_{args.split}{suffix}"

    builder = DatasetBuilder(
        cfg,
        dry_run=args.dry_run,
        build_workers=args.build_workers,
        shard_id=args.shard_id,
        n_shards=args.n_shards,
    )
    builder.run()


def cmd_stats(cfg, args):
    from src.pipeline.compute_stats import compute_stats
    log.info("=== Stage 3: Compute Stats ===")
    
    output_dir = Path(cfg.storage.output_dir)
    
    if args.split and not args.test_mode:
        base_name = Path(cfg.storage.output_filename).stem
        suffix = Path(cfg.storage.output_filename).suffix
        dataset_path = output_dir / f"{base_name}_{args.split}{suffix}"
        stats_path = output_dir / f"stats_{args.split}.json"
    else:
        dataset_path = output_dir / cfg.storage.output_filename
        stats_path = output_dir / "stats.json"
    
    log.info(f"Computing stats for: {dataset_path}")
    compute_stats(dataset_path, stats_path)
    log.info(f"stats.json written to: {stats_path}")


def cmd_all(cfg, args):
    cmd_download(cfg, args)
    cmd_build(cfg, args)
    cmd_stats(cfg, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="CorrDiff Regional HRRR Dataset Pipeline V2 (32×32→256×256)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stage",
        choices=["download", "build", "stats", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: configs/pipeline_config.yaml)",
    )

    # Test mode
    parser.add_argument(
        "--test_mode", action="store_true",
        help=(
            "Run a 10-day pilot (2015-01-01 to 2015-01-10) to validate the "
            "pipeline before launching the full dataset build."
        ),
    )

    # Train/test split
    parser.add_argument(
        "--split", choices=["train", "test"], default=None,
        help=(
            "Build train set (2015-2023, 500k) or test set (2024-2025, 50k). "
            "Omit for custom date range."
        ),
    )

    # Download args
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Parallel CDS API workers (default: 5, max recommended: 10)",
    )
    parser.add_argument(
        "--skip_pressure", action="store_true",
        help="Skip pressure-level ERA5 download",
    )

    # Date range overrides
    parser.add_argument(
        "--date_start", default=None,
        help="Override config date_start, e.g. 2015-06-01 (ignored if --split or --test_mode)",
    )
    parser.add_argument(
        "--date_end", default=None,
        help="Override config date_end, e.g. 2015-06-30 (ignored if --split or --test_mode)",
    )

    # Build args
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Use synthetic HRRR data -- no S3 calls (for local testing)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Override n_samples (highest priority)",
    )
    parser.add_argument(
        "--samples_per_ts", type=int, default=None,
        help="Override samples_per_ts",
    )
    parser.add_argument(
        "--build_workers", type=int, default=20,
        help="Parallel HRRR download threads for build stage (default: 20)",
    )

    # Shard args (for parallel builds)
    parser.add_argument(
        "--shard_id", type=int, default=None,
        help="Shard index (0-based). Requires --n_shards.",
    )
    parser.add_argument(
        "--n_shards", type=int, default=None,
        help="Total number of shards. Requires --shard_id.",
    )

    args = parser.parse_args()

    # Validate shard args
    if (args.shard_id is None) != (args.n_shards is None):
        parser.error("--shard_id and --n_shards must be used together")
    if args.shard_id is not None:
        if args.shard_id < 0 or args.shard_id >= args.n_shards:
            parser.error(f"--shard_id must be 0..{args.n_shards - 1}")
    
    cfg = load_config(args.config)
    dispatch = {
        "download": cmd_download,
        "build":    cmd_build,
        "stats":    cmd_stats,
        "all":      cmd_all,
    }
    dispatch[args.stage](cfg, args)

if __name__ == "__main__":
    main()
