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


def inspect_trace_libraries(db: wrds.Connection, logger):
    logger.info("Inspecting likely TRACE libraries...")

    libraries = db.list_libraries()
    candidates = [
        lib for lib in libraries
        if any(x in lib.lower() for x in ["trace", "finra", "bond"])
    ]
    logger.info(f"Candidate TRACE-related libraries: {candidates}")

    for lib in candidates:
        try:
            tables = db.list_tables(library=lib)
            logger.info(f"Tables in {lib}: {tables[:100]}")
        except Exception as e:
            logger.warning(f"Could not inspect {lib}: {e}")


def preview_table(db: wrds.Connection, library: str, table: str, logger):
    logger.info(f"Previewing {library}.{table}...")
    try:
        df = db.get_table(library=library, table=table, obs=5)
        logger.info(f"Preview of {library}.{table}:\n{df.head()}")
        logger.info(f"Columns in {library}.{table}: {list(df.columns)}")
        return df
    except Exception as e:
        logger.warning(f"Could not preview {library}.{table}: {e}")
        return pd.DataFrame()


def pull_trace_sample(db, logger):
    logger.info("Pulling TRACE sample (Jan 2019)...")

    query = """
    SELECT
        trd_exctn_dt,
        cusip_id,
        rptd_pr,
        entrd_vol_qt,
        rpt_side_cd
    FROM wrdsapps_bondret.trace_enhanced_clean
    WHERE trd_exctn_dt >= '2019-01-01'
      AND trd_exctn_dt < '2019-02-01'
    """

    df = db.raw_sql(query, date_cols=["trd_exctn_dt"])

    logger.info(f"Pulled {len(df):,} TRACE sample rows.")
    return df


def main():
    logger = get_logger("pull_trace", log_file="logs/pull_trace.log")
    paths = load_yaml("config/paths.yaml")

    raw_trace_dir = Path(paths["raw_trace"])
    ensure_dir(raw_trace_dir)

    logger.info("Starting TRACE discovery.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    inspect_trace_libraries(db, logger)

    logger.info("After reviewing the log, set candidate TRACE tables explicitly.")
    
    df = pull_trace_sample(db, logger)

    outpath = raw_trace_dir / "trace_sample_2019_01.parquet"
    save_parquet(df, outpath)

    logger.info(f"Saved TRACE sample to {outpath}")

    logger.info("Running checks...")
    logger.info(f"Date range: {df['trd_exctn_dt'].min()} → {df['trd_exctn_dt'].max()}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Unique bonds: {df['cusip_id'].nunique():,}")
    logger.info(f"Missing price: {df['rptd_pr'].isna().mean():.2%}")
    logger.info(f"Missing volume: {df['entrd_vol_qt'].isna().mean():.2%}")

if __name__ == "__main__":
    main()