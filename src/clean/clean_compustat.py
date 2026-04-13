from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def clean_compustat(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Starting Compustat cleaning...")
    df = df.copy()

    raw_rows = len(df)
    logger.info(f"Initial rows: {raw_rows:,}")

    df["gvkey"] = df["gvkey"].astype("string").str.strip()
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["tic"] = df["tic"].astype("string").str.strip()
    df["cusip"] = df["cusip"].astype("string").str.strip()
    df["conm"] = df["conm"].astype("string").str.strip()
    df["fic"] = df["fic"].astype("string").str.strip()

    numeric_cols = [
        "fyearq", "fqtr", "fyr",
        "atq", "ltq", "dlttq", "dlcq", "cheq", "actq", "lctq",
        "saleq", "revtq", "oibdpq", "ibq", "xintq",
        "capxy", "ppentq",
        "ceqq", "seqq", "cshoq", "prccq", "mkvaltq"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing core identifiers / dates
    before = len(df)
    df = df.dropna(subset=["gvkey", "datadate"])
    logger.info(f"Dropped missing gvkey/datadate: {before - len(df):,}")

    # Restrict sample window to match cleaned FISD working window
    before = len(df)
    df = df[(df["datadate"] >= "2005-01-01") & (df["datadate"] <= "2025-12-31")]
    logger.info(f"Dropped outside sample window: {before - len(df):,}")


    # Deduplicate firm-quarter
    before = len(df)
    df = df.sort_values(["gvkey", "datadate"])
    df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="last")
    logger.info(f"Dropped duplicate gvkey-datadate rows: {before - len(df):,}")


    # Basic accounting sanity filters
    before = len(df)
    df = df[df["atq"].isna() | (df["atq"] > 0)]
    logger.info(f"Dropped non-positive atq rows: {before - len(df):,}")


    # Construct core controls

    # Size
    df["size_log_assets"] = np.where(df["atq"] > 0, np.log(df["atq"]), np.nan)

    # Leverage
    df["debt_total"] = df["dlttq"].fillna(0) + df["dlcq"].fillna(0)
    df["leverage"] = np.where(df["atq"] > 0, df["debt_total"] / df["atq"], np.nan)

    # Cash ratio
    df["cash_ratio"] = np.where(df["atq"] > 0, df["cheq"] / df["atq"], np.nan)

    # Profitability
    df["profitability_oibdp"] = np.where(df["atq"] > 0, df["oibdpq"] / df["atq"], np.nan)
    df["profitability_ib"] = np.where(df["atq"] > 0, df["ibq"] / df["atq"], np.nan)

    # Tangibility
    df["tangibility"] = np.where(df["atq"] > 0, df["ppentq"] / df["atq"], np.nan)

    # Market equity
    df["market_equity"] = df["cshoq"] * df["prccq"]

    # Market-to-book style proxy
    # Use market equity + debt over assets as a rough q-style measure
    df["mtb_proxy"] = np.where(
        df["atq"] > 0,
        (df["market_equity"].fillna(0) + df["debt_total"].fillna(0)) / df["atq"],
        np.nan
    )

    # Interest coverage
    df["interest_coverage"] = np.where(
        df["xintq"] > 0,
        df["oibdpq"] / df["xintq"],
        np.nan
    )

    # Net working capital ratio
    df["nwc_ratio"] = np.where(
        df["atq"] > 0,
        (df["actq"].fillna(0) - df["lctq"].fillna(0)) / df["atq"],
        np.nan
    )

    def winsorize(series, lower=0.01, upper=0.99):
        return series.clip(
            lower=series.quantile(lower),
            upper=series.quantile(upper)
        )
    
    # Apply winsorization
    for col in [
        "leverage",
        "cash_ratio",
        "profitability_oibdp",
        "tangibility",
        "mtb_proxy"
    ]:
        if col in df.columns:
            df[col + "_w"] = winsorize(df[col])

    # Quarter-end monthly timestamp for later monthly expansion
    df["quarter_month"] = df["datadate"].dt.to_period("M").dt.to_timestamp("M")


    # Keep core columns
    keep_cols = [
        "gvkey",
        "datadate",
        "quarter_month",
        "fyearq",
        "fqtr",
        "fyr",
        "tic",
        "cusip",
        "conm",
        "fic",

        "atq",
        "ltq",
        "dlttq",
        "dlcq",
        "cheq",
        "actq",
        "lctq",
        "saleq",
        "revtq",
        "oibdpq",
        "ibq",
        "xintq",
        "capxy",
        "ppentq",
        "ceqq",
        "seqq",
        "cshoq",
        "prccq",
        "mkvaltq",

        "debt_total",
        "size_log_assets",
        "leverage",
        "cash_ratio",
        "profitability_oibdp",
        "profitability_ib",
        "tangibility",
        "market_equity",
        "mtb_proxy",
        "interest_coverage",
        "nwc_ratio",

        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "mtb_proxy_w",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()


    # Sort and reset
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)


    # Diagnostics
    final_rows = len(df)
    logger.info(f"Final clean rows: {final_rows:,}")
    logger.info(f"Rows dropped total: {raw_rows - final_rows:,}")
    logger.info(f"Percent dropped total: {(raw_rows - final_rows) / raw_rows:.2%}")
    logger.info(f"Date range: {df['datadate'].min()} → {df['datadate'].max()}")
    logger.info(f"Unique gvkeys: {df['gvkey'].nunique():,}")

    logger.info(f"Missing atq: {df['atq'].isna().mean():.2%}")
    logger.info(f"Missing leverage: {df['leverage'].isna().mean():.2%}")
    logger.info(f"Missing cash_ratio: {df['cash_ratio'].isna().mean():.2%}")
    logger.info(f"Missing profitability_oibdp: {df['profitability_oibdp'].isna().mean():.2%}")
    logger.info(f"Missing tangibility: {df['tangibility'].isna().mean():.2%}")
    logger.info(f"Missing market_equity: {df['market_equity'].isna().mean():.2%}")

    logger.info(
        f"Leverage quantiles: {df['leverage'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"Cash ratio quantiles: {df['cash_ratio'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"Profitability quantiles: {df['profitability_oibdp'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"Leverage_w quantiles: {df['leverage_w'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"Cash ratio_w quantiles: {df['cash_ratio_w'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"Profitability_w quantiles: {df['profitability_oibdp_w'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    return df


def main() -> None:
    logger = get_logger("clean_compustat", log_file="logs/clean_compustat.log")
    paths = load_yaml("config/paths.yaml")

    raw_path = Path(paths["raw_compustat"]) / "compustat_q_raw.parquet"
    out_dir = Path(paths["data_processed"]) / "compustat"
    out_path = out_dir / "compustat_q_clean.parquet"

    ensure_dir(out_dir)

    logger.info(f"Reading raw Compustat from {raw_path}")
    df_raw = read_parquet(raw_path)

    df_clean = clean_compustat(df_raw, logger)

    save_parquet(df_clean, out_path)
    logger.info(f"Saved cleaned Compustat to {out_path}")


if __name__ == "__main__":
    main()