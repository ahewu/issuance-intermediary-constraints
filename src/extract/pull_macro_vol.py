from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv

from src.utils.io import save_parquet, load_yaml, ensure_dir
from src.utils.logging_utils import get_logger


def connect_wrds() -> wrds.Connection:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")

    if not username:
        raise ValueError("WRDS_USERNAME not found in .env")

    return wrds.Connection(wrds_username=username)


def inspect_macro_libraries(db: wrds.Connection, logger):
    logger.info("Inspecting likely macro libraries...")

    libraries = db.list_libraries()
    candidates = [
        lib for lib in libraries
        if any(x in lib.lower() for x in ["crsp", "frb", "fred", "ff"])
    ]
    logger.info(f"Candidate macro libraries: {candidates}")

    for lib in candidates:
        try:
            tables = db.list_tables(library=lib)
            logger.info(f"Tables in {lib}: {tables[:50]}")
        except Exception as e:
            logger.warning(f"Could not inspect {lib}: {e}")


def preview_frb_rates(db, logger):
    logger.info("Previewing columns for frb.rates_daily...")
    df_daily = db.get_table(library="frb", table="rates_daily", obs=5)
    logger.info(f"Preview of frb.rates_daily:\n{df_daily.head()}")
    logger.info(f"Columns in frb.rates_daily: {list(df_daily.columns)}")

    logger.info("Previewing columns for frb.rates_monthly...")
    df_monthly = db.get_table(library="frb", table="rates_monthly", obs=5)
    logger.info(f"Preview of frb.rates_monthly:\n{df_monthly.head()}")
    logger.info(f"Columns in frb.rates_monthly: {list(df_monthly.columns)}")


def pull_rates_daily(db, logger):
    logger.info("Pulling FRB daily Treasury rates...")

    query = """
    SELECT
        date,
        dgs10,
        dgs2
    FROM frb.rates_daily
    WHERE dgs10 IS NOT NULL
    """

    df = db.raw_sql(query, date_cols=["date"])

    logger.info(f"Pulled {len(df):,} daily observations.")
    return df


def build_monthly_vol(df, logger):
    logger.info("Constructing monthly volatility panel...")

    df = df.sort_values("date")

    # daily change in 10Y yield
    df["d_dgs10"] = df["dgs10"].diff()

    # month index
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = (
        df.groupby("month")
        .agg(
            ust10y_eom=("dgs10", "last"),
            ust10y_avg=("dgs10", "mean"),
            ust2y_eom=("dgs2", "last"),
            vol_main=("d_dgs10", "std"),
        )
        .reset_index()
    )

    # term spread
    monthly["term_spread"] = monthly["ust10y_eom"] - monthly["ust2y_eom"]

    logger.info(f"Constructed {len(monthly):,} monthly observations.")
    return monthly


def main():
    logger = get_logger("pull_macro_vol", log_file="logs/pull_macro_vol.log")
    paths = load_yaml("config/paths.yaml")

    raw_move_dir = Path(paths["raw_move"])
    ensure_dir(raw_move_dir)

    logger.info("Starting macro/volatility pull.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    df_daily = pull_rates_daily(db, logger)

    df_monthly = build_monthly_vol(df_daily, logger)

    outpath = raw_move_dir / "macro_vol_raw.parquet"
    save_parquet(df_monthly, outpath)

    logger.info(f"Saved macro volatility data to {outpath}")

    logger.info("Running checks...")
    logger.info(f"Date range: {df_monthly['month'].min()} → {df_monthly['month'].max()}")
    logger.info(f"Rows: {len(df_monthly):,}")
    logger.info(f"Missing vol_main: {df_monthly['vol_main'].isna().mean():.2%}")


if __name__ == "__main__":
    main()