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


def list_fisd_tables(db: wrds.Connection, logger) -> list[str]:
    logger.info("Inspecting available WRDS libraries and FISD tables...")

    libraries = db.list_libraries()
    fisd_libraries = [lib for lib in libraries if "fisd" in lib.lower()]
    logger.info(f"FISD-related libraries found: {fisd_libraries}")

    all_tables = []
    for lib in fisd_libraries:
        try:
            tables = db.list_tables(library=lib)
            logger.info(f"Tables in {lib}: {tables}")
            all_tables.extend([f"{lib}.{t}" for t in tables])
        except Exception as e:
            logger.warning(f"Could not inspect library {lib}: {e}")

    return all_tables


def preview_table_columns(db: wrds.Connection, table_name: str, logger) -> pd.DataFrame:
    logger.info(f"Previewing columns for {table_name}...")
    try:
        df = db.get_table(
            library=table_name.split(".")[0],
            table=table_name.split(".")[1],
            obs=5,
        )
        logger.info(f"Preview of {table_name}:\n{df.head()}")
        logger.info(f"Columns in {table_name}: {list(df.columns)}")
        return df
    except Exception as e:
        logger.warning(f"Could not preview {table_name}: {e}")
        return pd.DataFrame()


def pull_fisd_issues(db, logger):
    logger.info("Pulling FISD mergedissue (core issuance data)...")

    query = """
    SELECT
        issue_id,
        issuer_id,
        complete_cusip,
        prospectus_issuer_name,

        offering_date,
        maturity,

        offering_amt,
        principal_amt,

        coupon,
        coupon_type,

        currency,
        foreign_currency,

        rule_144a,
        private_placement,

        asset_backed,
        convertible,
        putable,
        redeemable,

        bond_type,
        security_level

    FROM fisd.fisd_mergedissue
    WHERE offering_date IS NOT NULL
    """

    df = db.raw_sql(query)

    logger.info(f"Pulled {len(df):,} rows.")
    return df


def main():
    logger = get_logger("pull_fisd", log_file="logs/pull_fisd.log")
    paths = load_yaml("config/paths.yaml")

    raw_fisd_dir = Path(paths["raw_fisd"])
    ensure_dir(raw_fisd_dir)

    logger.info("Starting FISD pull.")

    db = connect_wrds()
    logger.info("Connected to WRDS successfully.")

    all_fisd_tables = list_fisd_tables(db, logger)

    if not all_fisd_tables:
        raise ValueError("No FISD-related libraries/tables found on WRDS account.")

    candidate_tables = [
    "fisd.fisd_mergedissue",
    "fisd.fisd_issue",
    "fisd.issue_issuer",
    "fisd.fisd_issuer",
    "fisd.fisd_ratings"
    ]

    for tbl in candidate_tables:
        preview_table_columns(db, tbl, logger)

    
    logger.info(
        "After inspecting logs, set target library/table explicitly in this script."
    )

    target_library = "fisd"
    target_table = "fisd_mergedissue"

    if target_library is None or target_table is None:
        logger.info("No target FISD table selected yet. Metadata inspection complete.")
        return

    df = pull_fisd_issues(db, logger)

    outpath = raw_fisd_dir / "fisd_issues_raw.parquet"
    save_parquet(df, outpath)

    logger.info(f"Saved raw FISD data to {outpath}")
    logger.info("Running checks...")

    logger.info(f"Date range: {df['offering_date'].min()} → {df['offering_date'].max()}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Missing offering_amt: {df['offering_amt'].isna().mean():.2%}")
    rule_144a_share = (
        df["rule_144a"]
        .astype("string")
        .str.upper()
        .eq("Y")
        .mean()
    )

    logger.info(f"144A share: {rule_144a_share:.2%}")

if __name__ == "__main__":
    main()