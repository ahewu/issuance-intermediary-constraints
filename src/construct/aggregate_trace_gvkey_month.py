from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def prepare_fisd_link_base(fisd_linked: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Preparing FISD-linked bond base...")

    df = fisd_linked.copy()

    df["complete_cusip"] = df["complete_cusip"].astype("string").str.strip()
    df["gvkey"] = df["gvkey"].astype("string").str.strip()

    # Keep only matched issues
    before = len(df)
    df = df[df["gvkey"].notna()].copy()
    logger.info(f"Dropped unmatched FISD issues: {before - len(df):,}")

    # One cusip -> one gvkey preferred
    # If duplicates exist, keep the first deterministically
    keep_cols = [
        "complete_cusip",
        "gvkey",
        "issuer_id",
        "issue_id",
        "issue_date",
        "issue_month",
        "amount_issued",
        "link_method",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.sort_values(["complete_cusip", "issue_date"], na_position="last")
    df = df.drop_duplicates(subset=["complete_cusip"], keep="first")

    logger.info(f"Usable FISD-linked cusips: {len(df):,}")
    logger.info(f"Unique gvkeys in FISD-linked base: {df['gvkey'].nunique():,}")

    return df


def merge_trace_to_fisd(trace_bond_month: pd.DataFrame, fisd_base: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Merging TRACE bond-month to linked FISD base...")

    trace = trace_bond_month.copy()
    trace["cusip_id"] = trace["cusip_id"].astype("string").str.strip()

    merged = trace.merge(
        fisd_base,
        left_on="cusip_id",
        right_on="complete_cusip",
        how="left",
        suffixes=("", "_fisd"),
    )

    logger.info(f"TRACE bond-month rows: {len(trace):,}")
    logger.info(f"Rows with gvkey after merge: {merged['gvkey'].notna().sum():,}")
    logger.info(f"TRACE->gvkey link rate: {merged['gvkey'].notna().mean():.2%}")

    return merged


def aggregate_to_gvkey_month(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Aggregating TRACE bond-month to gvkey-month...")

    df = df.copy()

    # Keep only rows with gvkey
    before = len(df)
    df = df[df["gvkey"].notna()].copy()
    logger.info(f"Dropped TRACE rows without gvkey: {before - len(df):,}")

    # Weighted helpers
    df["weight_dollar_volume"] = df["dollar_volume"].fillna(0)

    def weighted_mean(x: pd.Series, w: pd.Series) -> float:
        mask = x.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return np.average(x[mask], weights=w[mask])

    grouped = (
        df.groupby(["gvkey", "month"], as_index=False)
        .agg(
            bond_count_traded=("cusip_id", "nunique"),
            trade_count_sum=("trade_count", "sum"),
            par_volume_sum=("par_volume", "sum"),
            dollar_volume_sum=("dollar_volume", "sum"),
            avg_price_mean=("avg_price", "mean"),
            median_price_mean=("median_price", "mean"),
            price_std_mean=("price_std", "mean"),
            price_range_mean=("price_range", "mean"),
            amihud_proxy_mean=("amihud_proxy", "mean"),
        )
    )

    # Weighted averages
    weighted = (
        df.groupby(["gvkey", "month"])
        .apply(
            lambda g: pd.Series({
                "avg_price_weighted": weighted_mean(g["avg_price"], g["weight_dollar_volume"]),
                "price_std_weighted": weighted_mean(g["price_std"], g["weight_dollar_volume"]),
                "price_range_weighted": weighted_mean(g["price_range"], g["weight_dollar_volume"]),
                "amihud_proxy_weighted": weighted_mean(g["amihud_proxy"], g["weight_dollar_volume"]),
            })
        )
        .reset_index()
    )

    out = grouped.merge(weighted, on=["gvkey", "month"], how="left")

    # Winsorized liquidity proxies for later regressions
    if "amihud_proxy_mean" in out.columns:
        lo = out["amihud_proxy_mean"].quantile(0.01)
        hi = out["amihud_proxy_mean"].quantile(0.99)
        out["amihud_proxy_mean_w"] = out["amihud_proxy_mean"].clip(lo, hi)

    if "amihud_proxy_weighted" in out.columns:
        lo = out["amihud_proxy_weighted"].quantile(0.01)
        hi = out["amihud_proxy_weighted"].quantile(0.99)
        out["amihud_proxy_weighted_w"] = out["amihud_proxy_weighted"].clip(lo, hi)

    # Log volume controls
    out["log_dollar_volume_sum"] = np.where(
        out["dollar_volume_sum"] > 0,
        np.log(out["dollar_volume_sum"]),
        np.nan,
    )
    out["log_trade_count_sum"] = np.where(
        out["trade_count_sum"] > 0,
        np.log(out["trade_count_sum"]),
        np.nan,
    )

    out = out.sort_values(["gvkey", "month"]).reset_index(drop=True)

    logger.info(f"Constructed gvkey-month rows: {len(out):,}")
    logger.info(f"Unique gvkeys in TRACE gvkey-month: {out['gvkey'].nunique():,}")

    return out


def main() -> None:
    logger = get_logger(
        "aggregate_trace_gvkey_month",
        log_file="logs/aggregate_trace_gvkey_month.log",
    )
    paths = load_yaml("config/paths.yaml")

    trace_path = Path(paths["data_intermediate"]) / "trace" / "trace_bond_month.parquet"
    fisd_linked_path = Path(paths["data_processed"]) / "fisd" / "fisd_issues_linked.parquet"

    out_dir = Path(paths["data_intermediate"]) / "trace"
    out_path = out_dir / "trace_gvkey_month.parquet"

    ensure_dir(out_dir)

    logger.info(f"Reading TRACE bond-month from {trace_path}")
    trace_bond_month = read_parquet(trace_path)

    logger.info(f"Reading linked FISD from {fisd_linked_path}")
    fisd_linked = read_parquet(fisd_linked_path)

    fisd_base = prepare_fisd_link_base(fisd_linked, logger)
    merged = merge_trace_to_fisd(trace_bond_month, fisd_base, logger)
    gvkey_month = aggregate_to_gvkey_month(merged, logger)

    logger.info("Running final diagnostics...")
    logger.info(f"Final gvkey-month rows: {len(gvkey_month):,}")
    logger.info(f"Unique gvkeys: {gvkey_month['gvkey'].nunique():,}")
    logger.info(f"Date range: {gvkey_month['month'].min()} → {gvkey_month['month'].max()}")
    logger.info(f"Median bond_count_traded: {gvkey_month['bond_count_traded'].median():,.0f}")
    logger.info(f"Median trade_count_sum: {gvkey_month['trade_count_sum'].median():,.0f}")
    logger.info(f"Median dollar_volume_sum: {gvkey_month['dollar_volume_sum'].median():,.0f}")
    logger.info(f"Missing amihud_proxy_mean: {gvkey_month['amihud_proxy_mean'].isna().mean():.2%}")
    logger.info(f"Missing amihud_proxy_weighted: {gvkey_month['amihud_proxy_weighted'].isna().mean():.2%}")

    logger.info(
        f"bond_count_traded quantiles: {gvkey_month['bond_count_traded'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"trade_count_sum quantiles: {gvkey_month['trade_count_sum'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )
    logger.info(
        f"amihud_proxy_mean quantiles: {gvkey_month['amihud_proxy_mean'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()}"
    )

    save_parquet(gvkey_month, out_path)
    logger.info(f"Saved TRACE gvkey-month panel to {out_path}")


if __name__ == "__main__":
    main()