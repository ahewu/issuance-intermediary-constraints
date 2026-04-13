from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    return series.clip(
        lower=series.quantile(lower),
        upper=series.quantile(upper),
    )


def clean_macro(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Starting macro cleaning...")
    df = df.copy()

    raw_rows = len(df)
    logger.info(f"Initial rows: {raw_rows:,}")


    # Standardize types
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    numeric_cols = [
        "ust10y_eom",
        "ust10y_avg",
        "ust2y_eom",
        "term_spread",
        "vol_main",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # Drop missing month
    before = len(df)
    df = df.dropna(subset=["month"])
    logger.info(f"Dropped missing month rows: {before - len(df):,}")


    # Restrict to project sample window
    before = len(df)
    df = df[(df["month"] >= "2005-01-31") & (df["month"] <= "2025-12-31")]
    logger.info(f"Dropped outside sample window: {before - len(df):,}")


    # Deduplicate month
    before = len(df)
    df = df.sort_values("month").drop_duplicates(subset=["month"], keep="last")
    logger.info(f"Dropped duplicate months: {before - len(df):,}")


    # Rename core variables
    rename_map = {
        "vol_main": "rate_vol_10y",
    }
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)


    # Derived variables
    if "ust10y_eom" in df.columns and "ust2y_eom" in df.columns and "term_spread" not in df.columns:
        df["term_spread"] = df["ust10y_eom"] - df["ust2y_eom"]

    if "ust10y_eom" in df.columns:
        df["d_ust10y_eom"] = df["ust10y_eom"].diff()

    if "rate_vol_10y" in df.columns:
        df["log_rate_vol_10y"] = np.where(df["rate_vol_10y"] > 0, np.log(df["rate_vol_10y"]), np.nan)
        df["rate_vol_10y_w"] = winsorize(df["rate_vol_10y"])
        df["log_rate_vol_10y_w"] = winsorize(df["log_rate_vol_10y"])

    if "term_spread" in df.columns:
        df["term_spread_w"] = winsorize(df["term_spread"])


    # Keep core columns
    keep_cols = [
        "month",
        "rate_vol_10y",
        "rate_vol_10y_w",
        "log_rate_vol_10y",
        "log_rate_vol_10y_w",
        "ust10y_eom",
        "ust10y_avg",
        "ust2y_eom",
        "term_spread",
        "term_spread_w",
        "d_ust10y_eom",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()


    # Final sort
    df = df.sort_values("month").reset_index(drop=True)


    # Diagnostics
    final_rows = len(df)
    logger.info(f"Final clean rows: {final_rows:,}")
    logger.info(f"Rows dropped total: {raw_rows - final_rows:,}")
    logger.info(f"Percent dropped total: {(raw_rows - final_rows) / raw_rows:.2%}")
    logger.info(f"Date range: {df['month'].min()} → {df['month'].max()}")

    if "rate_vol_10y" in df.columns:
        logger.info(f"Missing rate_vol_10y: {df['rate_vol_10y'].isna().mean():.2%}")
        logger.info(
            f"rate_vol_10y quantiles: {df['rate_vol_10y'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
        )

    if "term_spread" in df.columns:
        logger.info(f"Missing term_spread: {df['term_spread'].isna().mean():.2%}")
        logger.info(
            f"term_spread quantiles: {df['term_spread'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
        )

    return df


def main() -> None:
    logger = get_logger("clean_macro", log_file="logs/clean_macro.log")
    paths = load_yaml("config/paths.yaml")

    raw_path = Path(paths["raw_move"]) / "macro_vol_raw.parquet"
    out_dir = Path(paths["data_processed"]) / "macro"
    out_path = out_dir / "macro_month_clean.parquet"

    ensure_dir(out_dir)

    logger.info(f"Reading raw macro data from {raw_path}")
    df_raw = read_parquet(raw_path)

    df_clean = clean_macro(df_raw, logger)

    save_parquet(df_clean, out_path)
    logger.info(f"Saved cleaned macro data to {out_path}")


if __name__ == "__main__":
    main()