from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils.io import load_yaml, read_parquet, ensure_dir
from src.utils.logging_utils import get_logger


def main():
    logger = get_logger("build_summary_stats")

    paths = load_yaml("config/paths.yaml")
    panel_path = Path(paths["data_processed"]) / "panels" / "firm_month_regression.parquet"
    out_dir = Path(paths["data_output"]) / "tables"
    ensure_dir(out_dir)

    logger.info(f"Reading panel from {panel_path}")
    df = read_parquet(panel_path)

    df = df[df["sample_regression_main"] == 1].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.sort_values("month")

    # Create lagged spread if spread exists
    if "credit_spread_w" in df.columns:
        df["credit_spread_w_lag"] = df["credit_spread_w"].shift(1)

    cols = [
        "issued",
        "log_total_issued_1p",
        "log_rate_vol_10y_w",
        "credit_spread_w",
        "credit_spread_w_lag",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "term_spread_w",
    ]

    # Keep only columns that actually exist
    cols = [c for c in cols if c in df.columns]

    df = df[cols].replace([float("inf"), float("-inf")], pd.NA)

    summary = df.describe(percentiles=[0.25, 0.5, 0.75]).T

    summary = summary.rename(columns={
        "count": "N",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "P25",
        "50%": "Median",
        "75%": "P75",
        "max": "Max",
    })

    rename_map = {
        "issued": "Issued (indicator)",
        "log_total_issued_1p": "Log issuance (1+)",
        "log_rate_vol_10y_w": "Rate volatility (log)",
        "credit_spread_w": "Credit spread",
        "credit_spread_w_lag": "Credit spread (lag)",
        "leverage_w": "Leverage",
        "cash_ratio_w": "Cash ratio",
        "profitability_oibdp_w": "Profitability",
        "tangibility_w": "Tangibility",
        "log_market_equity_w": "Log market equity",
        "term_spread_w": "Term spread",
    }
    summary.index = summary.index.map(lambda x: rename_map.get(x, x))

    summary = summary[["N", "Mean", "Std", "P25", "Median", "P75"]].round(4)

    out_csv = out_dir / "summary_stats.csv"
    out_tex = out_dir / "summary_stats.tex"

    summary.to_csv(out_csv)
    summary.to_latex(out_tex)

    logger.info(f"Saved summary stats to {out_csv}")
    logger.info(f"Saved LaTeX summary stats to {out_tex}")


if __name__ == "__main__":
    main()