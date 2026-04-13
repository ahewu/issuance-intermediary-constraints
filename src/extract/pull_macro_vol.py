from __future__ import annotations

import os
from pathlib import Path

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


def pull_frb_daily_treasury(db: wrds.Connection, logger) -> pd.DataFrame:
    logger.info("Pulling FRB daily Treasury rates...")

    query = """
    SELECT
        date,
        dgs10,
        dgs2
    FROM frb.rates_daily
    WHERE date >= '1962-01-01'
    ORDER BY date
    """

    df = db.raw_sql(query, date_cols=["date"])
    logger.info(f"Pulled {len(df):,} daily observations.")
    return df


def construct_monthly_vol_panel(df_daily: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Constructing monthly volatility panel...")

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["dgs10"] = pd.to_numeric(df["dgs10"], errors="coerce")
    df["dgs2"] = pd.to_numeric(df["dgs2"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    # daily 10y change
    df["d10"] = df["dgs10"].diff()

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            ust10y_eom=("dgs10", "last"),
            ust10y_avg=("dgs10", "mean"),
            ust2y_eom=("dgs2", "last"),
            vol_main=("d10", "std"),
        )
    )

    monthly["term_spread"] = monthly["ust10y_eom"] - monthly["ust2y_eom"]

    logger.info(f"Constructed {len(monthly):,} monthly observations.")
    return monthly


def pull_frb_monthly_credit_rates(db: wrds.Connection, logger) -> pd.DataFrame:
    logger.info("Pulling FRB monthly credit and Treasury rates...")

    query = """
    SELECT
        date,
        baa,
        aaa,
        gs10
    FROM frb.rates_monthly
    WHERE date >= '1962-01-01'
    ORDER BY date
    """

    df = db.raw_sql(query, date_cols=["date"])
    logger.info(f"Pulled {len(df):,} FRB monthly rate observations.")
    return df


def prepare_monthly_credit_rates(df_monthly_rates: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Preparing monthly credit-rate panel...")

    df = df_monthly_rates.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.rename(columns={"date": "month", "gs10": "gs10_monthly"})

    for col in ["baa", "aaa", "gs10_monthly"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Prepared monthly credit-rate rows: {len(df):,}")
    return df


def main() -> None:
    logger = get_logger("pull_macro_vol", log_file="logs/pull_macro_vol.log")
    paths = load_yaml("config/paths.yaml")

    raw_move_dir = Path(paths["raw_move"])
    ensure_dir(raw_move_dir)

    logger.info("Starting macro/volatility pull.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    # Daily Treasury rates -> monthly vol panel
    df_daily = pull_frb_daily_treasury(db, logger)
    df_monthly = construct_monthly_vol_panel(df_daily, logger)

    # Monthly FRB credit rates
    df_credit = pull_frb_monthly_credit_rates(db, logger)
    df_credit = prepare_monthly_credit_rates(df_credit, logger)

    # Merge
    df_out = df_monthly.merge(df_credit, on="month", how="left")

    out_path = raw_move_dir / "macro_vol_raw.parquet"
    save_parquet(df_out, out_path)

    logger.info(f"Saved macro volatility data to {out_path}")
    logger.info("Running checks...")
    logger.info(f"Date range: {df_out['month'].min()} → {df_out['month'].max()}")
    logger.info(f"Rows: {len(df_out):,}")
    logger.info(f"Missing vol_main: {df_out['vol_main'].isna().mean():.2%}")

    if "baa" in df_out.columns:
        logger.info(f"Missing baa: {df_out['baa'].isna().mean():.2%}")
    if "aaa" in df_out.columns:
        logger.info(f"Missing aaa: {df_out['aaa'].isna().mean():.2%}")


if __name__ == "__main__":
    main()