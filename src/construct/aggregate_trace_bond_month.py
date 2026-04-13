from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def process_trace_year(path: Path, logger) -> pd.DataFrame:
    logger.info(f"Reading TRACE year file: {path}")
    df = read_parquet(path)

    # Core type cleaning
    df = df.copy()
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"], errors="coerce")
    df["cusip_id"] = df["cusip_id"].astype("string").str.strip()
    df["rptd_pr"] = pd.to_numeric(df["rptd_pr"], errors="coerce")
    df["entrd_vol_qt"] = pd.to_numeric(df["entrd_vol_qt"], errors="coerce")

    # Drop unusable rows
    before = len(df)
    df = df.dropna(subset=["trd_exctn_dt", "cusip_id", "rptd_pr", "entrd_vol_qt"])
    logger.info(f"Dropped unusable rows: {before - len(df):,}")

    before = len(df)
    df = df[(df["rptd_pr"] > 0) & (df["entrd_vol_qt"] > 0)]
    logger.info(f"Dropped non-positive price/volume rows: {before - len(df):,}")

    # Month index
    df["month"] = df["trd_exctn_dt"].dt.to_period("M").dt.to_timestamp("M")

    # Dollar volume proxy
    df["dollar_volume"] = df["rptd_pr"] * df["entrd_vol_qt"]

    # Daily price change proxy at bond-month level later needs ordering by date
    # Here we build simple monthly aggregates first
    grouped = (
        df.groupby(["cusip_id", "month"], as_index=False)
        .agg(
            trade_count=("rptd_pr", "size"),
            par_volume=("entrd_vol_qt", "sum"),
            dollar_volume=("dollar_volume", "sum"),
            avg_price=("rptd_pr", "mean"),
            median_price=("rptd_pr", "median"),
            price_std=("rptd_pr", "std"),
            min_price=("rptd_pr", "min"),
            max_price=("rptd_pr", "max"),
        )
    )

    grouped["price_range"] = grouped["max_price"] - grouped["min_price"]

    # Simple Amihud-style proxy
    # Here using absolute price dispersion over dollar volume as a first-pass liquidity proxy
    grouped["amihud_proxy"] = np.where(
        grouped["dollar_volume"] > 0,
        grouped["price_range"] / grouped["dollar_volume"],
        np.nan,
    )

    logger.info(f"Constructed bond-month rows from {path.name}: {len(grouped):,}")
    return grouped


def combine_trace_years(trace_dir: Path, years: list[int], logger) -> pd.DataFrame:
    yearly_panels = []

    for year in years:
        path = trace_dir / f"trace_{year}.parquet"
        if not path.exists():
            logger.warning(f"Missing TRACE file for {year}: {path}")
            continue

        year_panel = process_trace_year(path, logger)
        year_panel["source_year"] = year
        yearly_panels.append(year_panel)

    if not yearly_panels:
        raise ValueError("No TRACE yearly files found for aggregation.")

    combined = pd.concat(yearly_panels, ignore_index=True)
    logger.info(f"Combined bond-month rows: {len(combined):,}")

    # Deduplicate just in case
    before = len(combined)
    combined = combined.sort_values(["cusip_id", "month", "source_year"])
    combined = combined.drop_duplicates(subset=["cusip_id", "month"], keep="last")
    logger.info(f"Dropped duplicate cusip-month rows: {before - len(combined):,}")

    combined = combined.drop(columns=["source_year"], errors="ignore")
    combined = combined.sort_values(["cusip_id", "month"]).reset_index(drop=True)

    return combined


def main() -> None:
    logger = get_logger(
        "aggregate_trace_bond_month",
        log_file="logs/aggregate_trace_bond_month.log",
    )
    paths = load_yaml("config/paths.yaml")

    trace_dir = Path(paths["raw_trace"])
    out_dir = Path(paths["data_intermediate"]) / "trace"
    out_path = out_dir / "trace_bond_month.parquet"

    ensure_dir(out_dir)

    years = list(range(2010, 2026))

    logger.info("Starting TRACE bond-month aggregation...")
    df = combine_trace_years(trace_dir, years, logger)

    logger.info("Running final diagnostics...")
    logger.info(f"Final bond-month rows: {len(df):,}")
    logger.info(f"Unique bonds: {df['cusip_id'].nunique():,}")
    logger.info(f"Date range: {df['month'].min()} → {df['month'].max()}")
    logger.info(f"Median trade_count: {df['trade_count'].median():,.0f}")
    logger.info(f"Median par_volume: {df['par_volume'].median():,.0f}")
    logger.info(f"Missing amihud_proxy: {df['amihud_proxy'].isna().mean():.2%}")

    logger.info(
        f"trade_count quantiles: {df['trade_count'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"par_volume quantiles: {df['par_volume'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"amihud_proxy quantiles: {df['amihud_proxy'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )

    save_parquet(df, out_path)
    logger.info(f"Saved TRACE bond-month panel to {out_path}")


if __name__ == "__main__":
    main()