from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def clean_fisd(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Starting FISD cleaning...")
    df = df.copy()

    raw_rows = len(df)
    logger.info(f"Initial rows: {raw_rows:,}")

    # Standardize key columns
    df["issue_date"] = pd.to_datetime(df["offering_date"], errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["amount_issued"] = pd.to_numeric(df["offering_amt"], errors="coerce") * 1000
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")

    # Normalize strings
    string_cols = [
        "currency",
        "coupon_type",
        "rule_144a",
        "private_placement",
        "bond_type",
        "security_level",
        "complete_cusip",
        "prospectus_issuer_name",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Drop missing core fields
    before = len(df)
    df = df.dropna(subset=["issue_date", "maturity_date", "amount_issued", "complete_cusip"])
    logger.info(f"Dropped missing core fields: {before - len(df):,}")

    before = len(df)
    df = df[(df["issue_date"] >= "2005-01-01") & (df["issue_date"] <= "2025-12-31")]
    logger.info(f"Dropped outside sample window: {before - len(df):,}")

    # Keep valid maturity structure
    before = len(df)
    df = df[df["maturity_date"] > df["issue_date"]]
    logger.info(f"Dropped invalid maturity dates: {before - len(df):,}")

    # Keep positive issue size
    before = len(df)
    df = df[df["amount_issued"] > 0]
    logger.info(f"Dropped non-positive issue sizes: {before - len(df):,}")

    # USD only
    # Diagnose currency fields
    logger.info(
        f"Top currency values: {df['currency'].astype('string').value_counts(dropna=False).head(20).to_dict()}"
    )
    if "foreign_currency" in df.columns:
        logger.info(
            f"foreign_currency values: {df['foreign_currency'].astype('string').value_counts(dropna=False).to_dict()}"
        )

    # USD / domestic-currency filter
    if "foreign_currency" in df.columns:
        before = len(df)
        df = df[~df["foreign_currency"].astype("string").str.upper().eq("Y")]
        logger.info(f"Dropped foreign_currency=Y issues: {before - len(df):,}")
    else:
        before = len(df)
        df = df[df["currency"].astype("string").str.upper().str.contains("USD|DOLLAR", na=False)]
        logger.info(f"Dropped non-USD issues: {before - len(df):,}")

     # before = len(df)
     # df = df[df["coupon_type"].str.upper().str.contains("FIX", na=False)]
     # logger.info(f"Dropped non-fixed-coupon issues: {before - len(df):,}")

    # Drop obvious non-corporate / structured products where possible 
    # conservative for now: remove asset-backed, convertibles, preferred securities, perpetuals
    for flag_col in ["asset_backed", "convertible", "preferred_security", "perpetual"]:
        if flag_col in df.columns:
            before = len(df)
            df = df[~df[flag_col].astype("string").str.upper().eq("Y")]
            logger.info(f"Dropped {flag_col}=Y: {before - len(df):,}")
    
    before = len(df)
    df = df[df["coupon_type"] == "F"]
    logger.info(f"Dropped non-fixed coupon issues: {before - len(df):,}")

    before = len(df)
    df = df[df["amount_issued"] >= 1e6]  # $1M cutoff
    logger.info(f"Dropped tiny issues (<$1M): {before - len(df):,}")

    before = len(df)
    df = df[df["amount_issued"] <= 1e10]  # $10B cap
    logger.info(f"Dropped extreme outliers (>$10B): {before - len(df):,}")

    # Construct useful flags
    df["is_144a"] = df["rule_144a"].astype("string").str.upper().eq("Y")
    df["is_private_placement"] = df["private_placement"].astype("string").str.upper().eq("Y")

    # Maturity at issuance
    df["maturity_years"] = (df["maturity_date"] - df["issue_date"]).dt.days / 365.25

    # Time indices
    df["issue_year"] = df["issue_date"].dt.year
    df["issue_month"] = df["issue_date"].dt.to_period("M").dt.to_timestamp("M")

    # Keep and rename core research columns
    keep_cols = [
        "issue_id",
        "issuer_id",
        "complete_cusip",
        "prospectus_issuer_name",
        "issue_date",
        "issue_month",
        "issue_year",
        "maturity_date",
        "maturity_years",
        "amount_issued",
        "coupon",
        "coupon_type",
        "currency",
        "rule_144a",
        "is_144a",
        "private_placement",
        "is_private_placement",
        "bond_type",
        "security_level",
        "asset_backed",
        "convertible",
        "putable",
        "redeemable",
        "preferred_security",
        "perpetual",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Deduplicate on issue identifier, then cusip as backup
    if "issue_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["issue_id"])
        logger.info(f"Dropped duplicate issue_id rows: {before - len(df):,}")
    else:
        before = len(df)
        df = df.drop_duplicates(subset=["complete_cusip"])
        logger.info(f"Dropped duplicate complete_cusip rows: {before - len(df):,}")

    # Sort
    sort_cols = [c for c in ["issue_date", "issuer_id", "issue_id"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    # Diagnostics: coupon type 
    if "coupon_type" in df.columns:
        logger.info(
            f"coupon_type values: {df['coupon_type'].astype('string').value_counts(dropna=False).head(20).to_dict()}"
        )

    # Diagnostics: issue size distribution 
    logger.info(
        f"amount_issued quantiles: {df['amount_issued'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )

    # Diagnostics: extreme values
    logger.info(
        f"Max issue size: {df['amount_issued'].max():,.0f}"
    )
    logger.info(
        f"Min positive issue size: {df[df['amount_issued'] > 0]['amount_issued'].min():,.0f}"
    )
    # Final checks
    final_rows = len(df)
    logger.info(f"Final clean rows: {final_rows:,}")
    logger.info(f"Rows dropped total: {raw_rows - final_rows:,}")
    logger.info(f"Percent dropped total: {(raw_rows - final_rows) / raw_rows:.2%}")
    logger.info(f"Issue date range: {df['issue_date'].min()} → {df['issue_date'].max()}")
    logger.info(f"Median issue size: {df['amount_issued'].median():,.0f}")
    logger.info(f"Mean 144A share: {df['is_144a'].mean():.2%}")
    logger.info(f"Mean private placement share: {df['is_private_placement'].mean():.2%}")
    logger.info(
        "Maturity years percentiles: "
        f"{df['maturity_years'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}"
    )

    return df


def main() -> None:
    logger = get_logger("clean_fisd", log_file="logs/clean_fisd.log")
    paths = load_yaml("config/paths.yaml")

    raw_path = Path(paths["raw_fisd"]) / "fisd_issues_raw.parquet"
    out_dir = Path(paths["data_processed"]) / "fisd"
    out_path = out_dir / "fisd_issues_clean.parquet"

    ensure_dir(out_dir)

    logger.info(f"Reading raw FISD from {raw_path}")
    df_raw = read_parquet(raw_path)

    df_clean = clean_fisd(df_raw, logger)

    save_parquet(df_clean, out_path)
    logger.info(f"Saved cleaned FISD to {out_path}")


if __name__ == "__main__":
    main()