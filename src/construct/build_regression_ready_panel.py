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


def build_regression_ready(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Building regression-ready panel...")
    df = df.copy()

    raw_rows = len(df)
    logger.info(f"Initial master-panel rows: {raw_rows:,}")

    # Standardize keys
    df["gvkey"] = df["gvkey"].astype("string").str.strip()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Time helpers
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["ym"] = df["month"].dt.strftime("%Y-%m")

    # Dependent variables
    df["issued"] = pd.to_numeric(df["issued"], errors="coerce").fillna(0)
    df["issue_count"] = pd.to_numeric(df["issue_count"], errors="coerce").fillna(0)
    df["total_issued"] = pd.to_numeric(df["total_issued"], errors="coerce").fillna(0)
    df["avg_issue_size"] = pd.to_numeric(df["avg_issue_size"], errors="coerce")

    df["log_total_issued_1p"] = np.log1p(df["total_issued"])
    df["log_issue_count_1p"] = np.log1p(df["issue_count"])

    # Optional intensive-margin variable among issuing months
    df["log_avg_issue_size"] = np.where(df["avg_issue_size"] > 0, np.log(df["avg_issue_size"]), np.nan)

    # Liquidity transforms
    for col in [
        "amihud_proxy_mean",
        "amihud_proxy_weighted",
        "amihud_proxy_mean_w",
        "amihud_proxy_weighted_w",
        "dollar_volume_sum",
        "trade_count_sum",
        "bond_count_traded",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "amihud_proxy_mean_w" in df.columns:
        df["log_amihud_mean_w"] = np.where(df["amihud_proxy_mean_w"] > 0, np.log(df["amihud_proxy_mean_w"]), np.nan)

    if "amihud_proxy_weighted_w" in df.columns:
        df["log_amihud_weighted_w"] = np.where(
            df["amihud_proxy_weighted_w"] > 0,
            np.log(df["amihud_proxy_weighted_w"]),
            np.nan,
        )

    if "dollar_volume_sum" in df.columns and "log_dollar_volume_sum" not in df.columns:
        df["log_dollar_volume_sum"] = np.where(df["dollar_volume_sum"] > 0, np.log(df["dollar_volume_sum"]), np.nan)

    if "trade_count_sum" in df.columns and "log_trade_count_sum" not in df.columns:
        df["log_trade_count_sum"] = np.where(df["trade_count_sum"] > 0, np.log(df["trade_count_sum"]), np.nan)

    # Firm controls
    if "market_equity" in df.columns:
        df["log_market_equity"] = np.where(df["market_equity"] > 0, np.log(df["market_equity"]), np.nan)

    # Optional extra winsorization at panel stage for raw variables if needed
    if "log_market_equity" in df.columns:
        df["log_market_equity_w"] = winsorize(df["log_market_equity"])

    # Macro transforms
    if "log_rate_vol_10y_w" not in df.columns and "rate_vol_10y" in df.columns:
        df["log_rate_vol_10y"] = np.where(df["rate_vol_10y"] > 0, np.log(df["rate_vol_10y"]), np.nan)
        df["log_rate_vol_10y_w"] = winsorize(df["log_rate_vol_10y"])

    # Sample flags
    required_controls = [
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
    ]
    required_controls = [c for c in required_controls if c in df.columns]

    liquidity_cols = [
        "log_amihud_weighted_w",
        "log_dollar_volume_sum",
        "log_trade_count_sum",
    ]
    liquidity_cols = [c for c in liquidity_cols if c in df.columns]

    macro_cols = [
        "log_rate_vol_10y_w",
        "term_spread_w",
    ]
    macro_cols = [c for c in macro_cols if c in df.columns]

    if required_controls:
        df["sample_has_controls"] = df[required_controls].notna().all(axis=1).astype(int)
    else:
        df["sample_has_controls"] = 0

    if liquidity_cols:
        df["sample_has_liquidity"] = df[liquidity_cols].notna().all(axis=1).astype(int)
    else:
        df["sample_has_liquidity"] = 0

    if macro_cols:
        df["sample_has_macro"] = df[macro_cols].notna().all(axis=1).astype(int)
    else:
        df["sample_has_macro"] = 0

    df["sample_main"] = (
        (df["sample_has_controls"] == 1)
        & (df["sample_has_liquidity"] == 1)
        & (df["sample_has_macro"] == 1)
    ).astype(int)


    # Conservative estimation sample window
    # TRACE starts in 2010; macro currently through 2025-02 in the processed file
    df["sample_window_main"] = (
        (df["month"] >= pd.Timestamp("2010-01-31"))
        & (df["month"] <= pd.Timestamp("2025-02-28"))
    ).astype(int)

    df["sample_regression_main"] = (
        (df["sample_main"] == 1)
        & (df["sample_window_main"] == 1)
    ).astype(int)


    # Final sort
    df = df.sort_values(["gvkey", "month"]).reset_index(drop=True)


    # Keep useful columns
    preferred_cols = [
        "gvkey",
        "month",
        "year",
        "month_num",
        "ym",

        # outcomes
        "issued",
        "issue_count",
        "total_issued",
        "avg_issue_size",
        "log_total_issued_1p",
        "log_issue_count_1p",
        "log_avg_issue_size",

        # liquidity
        "bond_count_traded",
        "trade_count_sum",
        "log_trade_count_sum",
        "dollar_volume_sum",
        "log_dollar_volume_sum",
        "amihud_proxy_mean",
        "amihud_proxy_mean_w",
        "amihud_proxy_weighted",
        "amihud_proxy_weighted_w",
        "log_amihud_mean_w",
        "log_amihud_weighted_w",

        # firm controls
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "market_equity",
        "log_market_equity_w",

        # macro
        "rate_vol_10y",
        "rate_vol_10y_w",
        "log_rate_vol_10y_w",
        "ust10y_eom",
        "ust2y_eom",
        "term_spread",
        "term_spread_w",

        "baa_treasury_spread",
        "baa_treasury_spread_w",
        "credit_spread",
        "credit_spread_w",

        # flags
        "sample_has_controls",
        "sample_has_liquidity",
        "sample_has_macro",
        "sample_main",
        "sample_window_main",
        "sample_regression_main",
    ]
    preferred_cols = [c for c in preferred_cols if c in df.columns]
    df = df[preferred_cols].copy()


    # Diagnostics
    logger.info(f"Final regression-ready rows: {len(df):,}")
    logger.info(f"Unique gvkeys: {df['gvkey'].nunique():,}")
    logger.info(f"Date range: {df['month'].min()} → {df['month'].max()}")
    logger.info(f"Issuance rate: {df['issued'].mean():.2%}")
    logger.info(f"Sample main share: {df['sample_main'].mean():.2%}")
    logger.info(f"Sample regression main share: {df['sample_regression_main'].mean():.2%}")

    if "log_amihud_weighted_w" in df.columns:
        logger.info(f"Missing log_amihud_weighted_w: {df['log_amihud_weighted_w'].isna().mean():.2%}")
    if "leverage_w" in df.columns:
        logger.info(f"Missing leverage_w: {df['leverage_w'].isna().mean():.2%}")
    if "log_rate_vol_10y_w" in df.columns:
        logger.info(f"Missing log_rate_vol_10y_w: {df['log_rate_vol_10y_w'].isna().mean():.2%}")

    return df


def main() -> None:
    logger = get_logger(
        "build_regression_ready_panel",
        log_file="logs/build_regression_ready_panel.log",
    )
    paths = load_yaml("config/paths.yaml")

    in_path = Path(paths["data_processed"]) / "panels" / "firm_month_master.parquet"
    out_dir = Path(paths["data_processed"]) / "panels"
    out_path = out_dir / "firm_month_regression.parquet"

    ensure_dir(out_dir)

    logger.info(f"Reading master panel from {in_path}")
    master = read_parquet(in_path)

    reg = build_regression_ready(master, logger)

    save_parquet(reg, out_path)
    logger.info(f"Saved regression-ready panel to {out_path}")


if __name__ == "__main__":
    main()