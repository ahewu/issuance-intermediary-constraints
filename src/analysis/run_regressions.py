from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils.io import load_yaml, ensure_dir, read_parquet
from src.utils.logging_utils import get_logger


# Helpers
def sanitize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def build_aggregate_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("month", as_index=False)
        .agg(
            liq_agg_mean=("liq_rank", "mean"),
            liq_agg_tradecount=("log_trade_count_sum", "mean"),
            liq_agg_dollarvol=("log_dollar_volume_sum", "mean"),
            firm_count=("gvkey", "nunique"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    agg["liq_agg_mean_lag1"] = agg["liq_agg_mean"].shift(1)
    return agg


def result_to_tidy(name: str, res, nobs: int) -> pd.DataFrame:
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "t_stat": res.tvalues.values,
        "p_value": res.pvalues.values,
    })
    out["model"] = name
    out["nobs"] = nobs
    return out


def fit_firm_fe_ols(df: pd.DataFrame, formula: str, cluster_col: str = "gvkey"):
    reg = df.copy()
    model = smf.ols(formula=formula, data=reg)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg[cluster_col]},
    )
    return res, reg


def fit_ts_ols(df: pd.DataFrame, y: str, xvars: list[str], hac_lags: int = 6):
    reg = df[[y] + xvars].dropna().copy()
    Y = reg[y]
    X = sm.add_constant(reg[xvars], has_constant="add")
    model = sm.OLS(Y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res, reg


# Main
def main() -> None:
    logger = get_logger("run_regressions", log_file="logs/run_regressions.log")
    paths = load_yaml("config/paths.yaml")

    panel_path = Path(paths["data_processed"]) / "panels" / "firm_month_regression.parquet"
    out_dir = Path(paths["data_processed"]) / "results"
    ensure_dir(out_dir)

    logger.info(f"Reading regression-ready panel from {panel_path}")
    df = read_parquet(panel_path)


    # Base sample
    df = df[df["sample_regression_main"] == 1].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str).str.strip()

    # Numeric cleanup
    numeric_cols = [
        "log_total_issued_1p",
        "issue_count",
        "total_issued",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
        "log_trade_count_sum",
        "log_dollar_volume_sum",
        "amihud_proxy_weighted_w",
    ]
    df = sanitize_columns(df, numeric_cols)

    # winsorize a few heavy-tailed controls again for safety
    for col in [
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
    ]:
        if col in df.columns:
            df[col] = winsorize(df[col])

    # stable aggregate-liquidity input
    df["liq_rank"] = df["amihud_proxy_weighted_w"].rank(pct=True)


    # Restrict to firms with variation in intensive-margin DV
    var_check = df.groupby("gvkey")["log_total_issued_1p"].std()
    valid_gvkeys = var_check[var_check > 1e-8].index
    df = df[df["gvkey"].isin(valid_gvkeys)].copy()


    # Aggregate liquidity time series
    liq_agg = build_aggregate_liquidity(df)
    df = df.merge(
        liq_agg[["month", "liq_agg_mean", "liq_agg_mean_lag1"]],
        on="month",
        how="left",
    )


    # Keep only columns needed for regressions

    keep_cols = [
        "gvkey",
        "month",
        "log_total_issued_1p",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
        "liq_agg_mean",
        "liq_agg_mean_lag1",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna().copy()

    logger.info(f"Regression sample rows after cleanup: {len(df):,}")
    logger.info(f"Regression sample firms after cleanup: {df['gvkey'].nunique():,}")

    model_results = []


    # 3.1 Baseline
    # Intensive margin only
    logger.info("Running 3.1 baseline FE OLS: log_total_issued_1p")
    formula_31 = (
        "log_total_issued_1p ~ log_rate_vol_10y_w + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_31, reg_31 = fit_firm_fe_ols(df, formula_31, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.1_baseline_log_total_issued", res_31, len(reg_31)))


    # 3.2 Liquidity AR(1)
    # Time series
    logger.info("Running 3.2 aggregate liquidity AR(1)")
    liq_ts = (
        df[["month", "log_rate_vol_10y_w", "term_spread_w"]]
        .drop_duplicates()
        .merge(liq_agg[["month", "liq_agg_mean", "liq_agg_mean_lag1"]], on="month", how="left")
        .sort_values("month")
    )
    res_32, reg_32 = fit_ts_ols(
        liq_ts,
        y="liq_agg_mean",
        xvars=["liq_agg_mean_lag1", "log_rate_vol_10y_w", "term_spread_w"],
        hac_lags=6,
    )
    model_results.append(result_to_tidy("3.2_liquidity_ar1", res_32, len(reg_32)))


    # 3.3 Mechanism: aggregate liquidity
    logger.info("Running 3.3 mechanism FE OLS: aggregate liquidity")
    formula_33 = (
        "log_total_issued_1p ~ log_rate_vol_10y_w + liq_agg_mean + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_33, reg_33 = fit_firm_fe_ols(df, formula_33, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.3_mechanism_aggregate_liquidity", res_33, len(reg_33)))


    # Save outputs
    results_df = pd.concat(model_results, ignore_index=True)
    results_path = out_dir / "regression_results_tidy.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved tidy regression results to {results_path}")

    summary = {
        "3.1_baseline_log_total_issued": {
            "nobs": int(len(reg_31)),
            "rsquared": float(res_31.rsquared),
        },
        "3.2_liquidity_ar1": {
            "nobs": int(len(reg_32)),
            "rsquared": float(res_32.rsquared),
        },
        "3.3_mechanism_aggregate_liquidity": {
            "nobs": int(len(reg_33)),
            "rsquared": float(res_33.rsquared),
        },
    }
    summary_path = out_dir / "regression_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary metadata to {summary_path}")


if __name__ == "__main__":
    main()