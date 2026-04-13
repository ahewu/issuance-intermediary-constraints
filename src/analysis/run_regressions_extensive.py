from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils.io import load_yaml, ensure_dir, read_parquet
from src.utils.logging_utils import get_logger


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


def fit_lpm(df: pd.DataFrame, formula: str, cluster_col: str = "gvkey"):
    reg = df.copy()
    model = smf.ols(formula=formula, data=reg)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg[cluster_col]},
    )
    return res, reg


def fit_logit(df: pd.DataFrame, formula: str, cluster_col: str = "gvkey"):
    reg = df.copy()
    model = smf.logit(formula=formula, data=reg)
    res = model.fit(
        disp=False,
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": reg[cluster_col]},
    )
    return res, reg


def main() -> None:
    logger = get_logger("run_regressions_extensive", log_file="logs/run_regressions_extensive.log")
    paths = load_yaml("config/paths.yaml")

    panel_path = Path(paths["data_processed"]) / "panels" / "firm_month_regression.parquet"
    out_dir = Path(paths["data_processed"]) / "results"
    ensure_dir(out_dir)

    logger.info(f"Reading regression-ready panel from {panel_path}")
    df = read_parquet(panel_path)

    # ----------------------------
    # Base sample
    # ----------------------------
    df = df[df["sample_regression_main"] == 1].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    df["issued"] = pd.to_numeric(df["issued"], errors="coerce").fillna(0).astype(float)

    # ----------------------------
    # Numeric cleanup
    # ----------------------------
    numeric_cols = [
        "issued",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
        "amihud_proxy_weighted_w",
    ]
    df = sanitize_columns(df, numeric_cols)

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

    # Aggregate liquidity
    df["liq_rank"] = df["amihud_proxy_weighted_w"].rank(pct=True)
    liq_agg = build_aggregate_liquidity(df)
    df = df.merge(
        liq_agg[["month", "liq_agg_mean", "liq_agg_mean_lag1"]],
        on="month",
        how="left",
    )

    # ----------------------------
    # Samples
    # ----------------------------
    keep_cols = [
        "gvkey",
        "month",
        "issued",
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

    logger.info(f"Extensive-margin sample rows: {len(df):,}")
    logger.info(f"Extensive-margin sample firms: {df['gvkey'].nunique():,}")
    logger.info(f"Issuance rate in sample: {df['issued'].mean():.2%}")

    model_results = []

    # ------------------------------------
    # 3.1 LPM with firm FE
    # ------------------------------------
    logger.info("Running 3.1 LPM with firm FE")
    formula_lpm_31 = (
        "issued ~ log_rate_vol_10y_w + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + "
        "term_spread_w + C(gvkey)"
    )
    res_lpm_31, reg_lpm_31 = fit_lpm(df, formula_lpm_31, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.1_lpm_fe_issued", res_lpm_31, len(reg_lpm_31)))

    # ------------------------------------
    # 3.3 LPM with firm FE + aggregate liquidity
    # ------------------------------------
    logger.info("Running 3.3 LPM with firm FE and aggregate liquidity")
    formula_lpm_33 = (
        "issued ~ log_rate_vol_10y_w + liq_agg_mean + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + "
        "term_spread_w + C(gvkey)"
    )
    res_lpm_33, reg_lpm_33 = fit_lpm(df, formula_lpm_33, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.3_lpm_fe_issued_aggregate_liquidity", res_lpm_33, len(reg_lpm_33)))

    # ------------------------------------
    # Robustness: pooled logit baseline
    # No firm FE here
    # ------------------------------------
    logger.info("Running pooled logit baseline")
    formula_logit_31 = (
        "issued ~ log_rate_vol_10y_w + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + term_spread_w"
    )
    res_logit_31, reg_logit_31 = fit_logit(df, formula_logit_31, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.1_logit_pooled_issued", res_logit_31, len(reg_logit_31)))

    # ------------------------------------
    # Robustness: pooled logit mechanism
    # ------------------------------------
    logger.info("Running pooled logit with aggregate liquidity")
    formula_logit_33 = (
        "issued ~ log_rate_vol_10y_w + liq_agg_mean + leverage_w + cash_ratio_w + "
        "profitability_oibdp_w + tangibility_w + log_market_equity_w + term_spread_w"
    )
    res_logit_33, reg_logit_33 = fit_logit(df, formula_logit_33, cluster_col="gvkey")
    model_results.append(result_to_tidy("3.3_logit_pooled_issued_aggregate_liquidity", res_logit_33, len(reg_logit_33)))

    # ------------------------------------
    # Save outputs
    # ------------------------------------
    results_df = pd.concat(model_results, ignore_index=True)
    results_path = out_dir / "regression_results_extensive_tidy.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved tidy extensive-margin regression results to {results_path}")

    summary = {
        "3.1_lpm_fe_issued": {
            "nobs": int(len(reg_lpm_31)),
            "rsquared": float(res_lpm_31.rsquared),
        },
        "3.3_lpm_fe_issued_aggregate_liquidity": {
            "nobs": int(len(reg_lpm_33)),
            "rsquared": float(res_lpm_33.rsquared),
        },
        "3.1_logit_pooled_issued": {
            "nobs": int(len(reg_logit_31)),
            "pseudo_rsquared": float(getattr(res_logit_31, "prsquared", np.nan)),
        },
        "3.3_logit_pooled_issued_aggregate_liquidity": {
            "nobs": int(len(reg_logit_33)),
            "pseudo_rsquared": float(getattr(res_logit_33, "prsquared", np.nan)),
        },
    }
    summary_path = out_dir / "regression_results_extensive_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved extensive-margin summary metadata to {summary_path}")


if __name__ == "__main__":
    main()