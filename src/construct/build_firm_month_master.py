from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


# Prepare Compustat monthly
def prepare_compustat_monthly(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Preparing full monthly Compustat panel...")

    df = df.copy()
    df["gvkey"] = df["gvkey"].astype("string").str.strip()
    df["month"] = pd.to_datetime(df["datadate"]).dt.to_period("M").dt.to_timestamp("M")

    keep_cols = [
        "gvkey",
        "month",
        "atq",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "market_equity",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Keep one row per gvkey-month at quarter-end
    df = df.sort_values(["gvkey", "month"])
    df = df.drop_duplicates(subset=["gvkey", "month"], keep="last")

    logger.info(f"Quarter-end Compustat rows before monthly expansion: {len(df):,}")

    # Build full monthly grid per gvkey
    month_ranges = (
        df.groupby("gvkey")["month"]
        .agg(month_min="min", month_max="max")
        .reset_index()
    )

    monthly_grids = []
    for row in month_ranges.itertuples(index=False):
        months = pd.date_range(start=row.month_min, end=row.month_max, freq="ME")
        monthly_grids.append(
            pd.DataFrame({
                "gvkey": row.gvkey,
                "month": months,
            })
        )

    full_monthly = pd.concat(monthly_grids, ignore_index=True)

    logger.info(f"Full gvkey-month grid rows: {len(full_monthly):,}")

    # Merge quarter-end observations onto full monthly grid
    full_monthly = full_monthly.merge(
        df,
        on=["gvkey", "month"],
        how="left",
    )

    # Forward-fill quarterly values within firm across months
    value_cols = [c for c in full_monthly.columns if c not in ["gvkey", "month"]]
    full_monthly = full_monthly.sort_values(["gvkey", "month"])
    full_monthly[value_cols] = full_monthly.groupby("gvkey")[value_cols].ffill()

    logger.info(f"Compustat monthly rows after expansion: {len(full_monthly):,}")
    logger.info(f"Compustat monthly unique gvkeys: {full_monthly['gvkey'].nunique():,}")

    return full_monthly


# Prepare issuance panel
def prepare_issuance_monthly(fisd: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Preparing issuance monthly panel...")

    df = fisd.copy()
    df = df[df["gvkey"].notna()].copy()

    df["month"] = pd.to_datetime(df["issue_date"]).dt.to_period("M").dt.to_timestamp("M")

    grouped = (
        df.groupby(["gvkey", "month"], as_index=False)
        .agg(
            issue_count=("issue_id", "size"),
            total_issued=("amount_issued", "sum"),
            avg_issue_size=("amount_issued", "mean"),
        )
    )

    # Binary issuance indicator
    grouped["issued"] = 1

    logger.info(f"Issuance gvkey-month rows: {len(grouped):,}")
    return grouped


# Merge everything
def build_master_panel(trace, comp, issuance, macro, logger):
    logger.info("Building unified firm-month panel...")

    df = trace.copy()

    # Merge Compustat
    df = df.merge(comp, on=["gvkey", "month"], how="left")

    # Merge issuance
    df = df.merge(issuance, on=["gvkey", "month"], how="left")

    df["issue_count"] = df["issue_count"].fillna(0)
    df["total_issued"] = df["total_issued"].fillna(0)
    df["issued"] = df["issued"].fillna(0)

    # Merge macro
    df = df.merge(macro, on="month", how="left")

    macro_cols = [c for c in macro.columns if c != "month"]
    df = df.sort_values(["gvkey", "month"])
    df[macro_cols] = df.groupby("gvkey")[macro_cols].ffill()

    # Final sort
    df = df.sort_values(["gvkey", "month"]).reset_index(drop=True)

    return df


def main():
    logger = get_logger("build_firm_month_master", log_file="logs/build_master.log")
    paths = load_yaml("config/paths.yaml")

    trace = read_parquet(Path(paths["data_intermediate"]) / "trace" / "trace_gvkey_month.parquet")
    comp = read_parquet(Path(paths["data_processed"]) / "compustat" / "compustat_q_clean.parquet")
    fisd = read_parquet(Path(paths["data_processed"]) / "fisd" / "fisd_issues_linked.parquet")
    macro = read_parquet(Path(paths["data_processed"]) / "macro" / "macro_month_clean.parquet")

    comp_m = prepare_compustat_monthly(comp, logger)
    issuance_m = prepare_issuance_monthly(fisd, logger)

    master = build_master_panel(trace, comp_m, issuance_m, macro, logger)

    # Diagnostics
    logger.info(f"Final panel rows: {len(master):,}")
    logger.info(f"Unique gvkeys: {master['gvkey'].nunique():,}")
    logger.info(f"Date range: {master['month'].min()} → {master['month'].max()}")

    logger.info(f"Issuance rate: {master['issued'].mean():.2%}")
    logger.info(f"Missing leverage: {master['leverage_w'].isna().mean():.2%}")
    logger.info(f"Missing macro: {master['rate_vol_10y'].isna().mean():.2%}")

    out_dir = Path(paths["data_processed"]) / "panels"
    ensure_dir(out_dir)

    out_path = out_dir / "firm_month_master.parquet"
    save_parquet(master, out_path)

    logger.info(f"Saved master panel to {out_path}")


if __name__ == "__main__":
    main()