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

def preview_compustat_columns(db, logger):
    logger.info("Previewing columns for comp.fundq...")
    df_preview = db.get_table(library="comp", table="fundq", obs=5)
    logger.info(f"Preview of comp.fundq:\n{df_preview.head()}")
    logger.info(f"Columns in comp.fundq: {list(df_preview.columns)}")
    return df_preview

def pull_compustat_quarterly(db: wrds.Connection, logger) -> pd.DataFrame:
    logger.info("Pulling Compustat quarterly fundamentals...")

    query = """
SELECT
    gvkey,
    datadate,
    fyearq,
    fqtr,
    fyr,
    tic,
    cusip,
    conm,
    fic,

    atq,
    ltq,
    dlttq,
    dlcq,
    cheq,
    actq,
    lctq,

    saleq,
    revtq,
    oibdpq,
    ibq,
    xintq,

    capxy,
    ppentq,

    ceqq,
    seqq,
    cshoq,
    prccq,
    mkvaltq

FROM comp.fundq
WHERE indfmt = 'INDL'
  AND datafmt = 'STD'
  AND popsrc = 'D'
  AND consol = 'C'
  AND datadate IS NOT NULL
"""

    df = db.raw_sql(query, date_cols=["datadate"])

    logger.info(f"Pulled {len(df):,} Compustat quarterly rows.")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def main():
    logger = get_logger("pull_compustat", log_file="logs/pull_compustat.log")
    paths = load_yaml("config/paths.yaml")

    raw_compustat_dir = Path(paths["raw_compustat"])
    ensure_dir(raw_compustat_dir)

    logger.info("Starting Compustat pull.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    df = pull_compustat_quarterly(db, logger)

    outpath = raw_compustat_dir / "compustat_q_raw.parquet"
    save_parquet(df, outpath)

    logger.info(f"Saved raw Compustat data to {outpath}")

    logger.info("Running checks...")
    logger.info(f"Date range: {df['datadate'].min()} → {df['datadate'].max()}")
    logger.info(f"Unique gvkeys: {df['gvkey'].nunique():,}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Missing atq: {df['atq'].isna().mean():.2%}")
    logger.info(f"Missing dlttq: {df['dlttq'].isna().mean():.2%}")
    logger.info(f"Missing dlcq: {df['dlcq'].isna().mean():.2%}")
    logger.info(f"Missing cheq: {df['cheq'].isna().mean():.2%}")
    logger.info(f"Missing oibdpq: {df['oibdpq'].isna().mean():.2%}")


if __name__ == "__main__":
    main()