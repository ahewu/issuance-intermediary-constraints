from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import wrds
from dotenv import load_dotenv

from src.utils.io import load_yaml, ensure_dir, save_parquet
from src.utils.logging_utils import get_logger


def connect_wrds() -> wrds.Connection:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")

    if not username:
        raise ValueError("WRDS_USERNAME not found in .env")

    return wrds.Connection(wrds_username=username)


def pull_trace_year(db: wrds.Connection, year: int, logger) -> pd.DataFrame:
    logger.info(f"Pulling TRACE for {year}...")

    query = f"""
    SELECT
        trd_exctn_dt,
        cusip_id,
        rptd_pr,
        entrd_vol_qt,
        rpt_side_cd
    FROM wrdsapps_bondret.trace_enhanced_clean
    WHERE trd_exctn_dt >= '{year}-01-01'
      AND trd_exctn_dt < '{year + 1}-01-01'
    """

    df = db.raw_sql(query, date_cols=["trd_exctn_dt"])

    logger.info(f"Pulled {len(df):,} rows for {year}.")
    return df


def validate_trace_year(df: pd.DataFrame, year: int, logger) -> None:
    logger.info(f"Running checks for {year}...")

    logger.info(f"{year} date range: {df['trd_exctn_dt'].min()} → {df['trd_exctn_dt'].max()}")
    logger.info(f"{year} rows: {len(df):,}")
    logger.info(f"{year} unique bonds: {df['cusip_id'].nunique():,}")
    logger.info(f"{year} missing price: {df['rptd_pr'].isna().mean():.2%}")
    logger.info(f"{year} missing volume: {df['entrd_vol_qt'].isna().mean():.2%}")

    if "rpt_side_cd" in df.columns:
        side_counts = (
            df["rpt_side_cd"]
            .astype("string")
            .fillna("MISSING")
            .value_counts()
            .to_dict()
        )
        logger.info(f"{year} rpt_side_cd distribution: {side_counts}")


def pull_trace_range(
    db: wrds.Connection,
    years: List[int],
    raw_trace_dir: Path,
    logger,
    skip_existing: bool = True,
) -> None:
    for year in years:
        outpath = raw_trace_dir / f"trace_{year}.parquet"

        if skip_existing and outpath.exists():
            logger.info(f"Skipping {year}; file already exists at {outpath}")
            continue

        df = pull_trace_year(db, year, logger)

        save_parquet(df, outpath)
        logger.info(f"Saved TRACE data for {year} to {outpath}")

        size_mb = outpath.stat().st_size / (1024 ** 2)
        logger.info(f"{year} file size: {size_mb:,.1f} MB")

        validate_trace_year(df, year, logger)


def main():
    logger = get_logger("pull_trace", log_file="logs/pull_trace.log")
    paths = load_yaml("config/paths.yaml")

    raw_trace_dir = Path(paths["raw_trace"])
    ensure_dir(raw_trace_dir)

    logger.info("Starting TRACE yearly pull.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    years = list(range(2010, 2026))
    pull_trace_range(
        db=db,
        years=years,
        raw_trace_dir=raw_trace_dir,
        logger=logger,
        skip_existing=True,
    )

    logger.info("TRACE yearly pull complete.")


if __name__ == "__main__":
    main()