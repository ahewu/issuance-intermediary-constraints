from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
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


def choose_spread_column(df: pd.DataFrame) -> str:
    candidates = [
        "credit_spread_w",
        "credit_spread",
        "baa_treasury_spread",
        "baa_aaa_spread",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
"No spread variable found. Add one of: credit_spread_w, credit_spread, "
"baa_treasury_spread, baa_aaa_spread to the regression-ready panel."
)


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


def main() -> None:
    logger = get_logger(
        "run_regressions_spread_pivot",
        log_file="logs/run_regressions_spread_pivot.log",
    )
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
    df["log_total_issued_1p"] = pd.to_numeric(df["log_total_issued_1p"], errors="coerce")

    spread_col = choose_spread_column(df)
    logger.info(f"Using spread variable: {spread_col}")

# ----------------------------
# Numeric cleanup
# ----------------------------
    numeric_cols = [
        "issued",
        "log_total_issued_1p",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
        spread_col,
    ]
    df = sanitize_columns(df, numeric_cols)

# winsorize core regressors for stability
    for col in [
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "log_rate_vol_10y_w",
        "term_spread_w",
        spread_col,
    ]:
        if col in df.columns:
            df[col] = winsorize(df[col])

# Key interaction
    df["vol_x_spread"] = df["log_rate_vol_10y_w"] * df[spread_col]

# ----------------------------
# Intensive-margin sample
# Keep firms with variation in log issuance
# ----------------------------
    intensive_df = df.copy()
    intensive_var = intensive_df.groupby("gvkey")["log_total_issued_1p"].std()
    valid_gvkeys = intensive_var[intensive_var > 1e-8].index
    intensive_df = intensive_df[intensive_df["gvkey"].isin(valid_gvkeys)].copy()

    intensive_keep = [
        "gvkey",
        "month",
        "log_total_issued_1p",
        "log_rate_vol_10y_w",
        spread_col,
        "vol_x_spread",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "term_spread_w",
    ]
    intensive_keep = [c for c in intensive_keep if c in intensive_df.columns]
    intensive_df = intensive_df[intensive_keep].dropna().copy()

    logger.info(f"Intensive-margin rows: {len(intensive_df):,}")
    logger.info(f"Intensive-margin firms: {intensive_df['gvkey'].nunique():,}")

# ----------------------------
# Extensive-margin sample
# ----------------------------
    extensive_df = df.copy()
    extensive_keep = [
        "gvkey",
        "month",
        "issued",
        "log_rate_vol_10y_w",
        spread_col,
        "vol_x_spread",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "term_spread_w",
    ]
    extensive_keep = [c for c in extensive_keep if c in extensive_df.columns]
    extensive_df = extensive_df[extensive_keep].dropna().copy()

    logger.info(f"Extensive-margin rows: {len(extensive_df):,}")
    logger.info(f"Extensive-margin firms: {extensive_df['gvkey'].nunique():,}")
    logger.info(f"Extensive-margin issuance rate: {extensive_df['issued'].mean():.2%}")

    model_results = []

# ------------------------------------
# Intensive baseline
# ------------------------------------
    logger.info("Running intensive baseline FE OLS")
    formula_int_base = (
        f"log_total_issued_1p ~ log_rate_vol_10y_w + {spread_col} + "
        "leverage_w + cash_ratio_w + profitability_oibdp_w + tangibility_w + "
        "log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_int_base, reg_int_base = fit_firm_fe_ols(intensive_df, formula_int_base, cluster_col="gvkey")
    model_results.append(result_to_tidy("pivot_intensive_baseline", res_int_base, len(reg_int_base)))

# ------------------------------------
# Intensive interaction
# ------------------------------------
    logger.info("Running intensive interaction FE OLS")
    formula_int_inter = (
        f"log_total_issued_1p ~ log_rate_vol_10y_w + {spread_col} + vol_x_spread + "
        "leverage_w + cash_ratio_w + profitability_oibdp_w + tangibility_w + "
        "log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_int_inter, reg_int_inter = fit_firm_fe_ols(intensive_df, formula_int_inter, cluster_col="gvkey")
    model_results.append(result_to_tidy("pivot_intensive_interaction", res_int_inter, len(reg_int_inter)))

# ------------------------------------
# Extensive baseline (LPM)
# ------------------------------------
    logger.info("Running extensive baseline FE LPM")
    formula_ext_base = (
        f"issued ~ log_rate_vol_10y_w + {spread_col} + "
        "leverage_w + cash_ratio_w + profitability_oibdp_w + tangibility_w + "
        "log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_ext_base, reg_ext_base = fit_firm_fe_ols(extensive_df, formula_ext_base, cluster_col="gvkey")
    model_results.append(result_to_tidy("pivot_extensive_baseline", res_ext_base, len(reg_ext_base)))

# ------------------------------------
# Extensive interaction (LPM)
# ------------------------------------
    logger.info("Running extensive interaction FE LPM")
    formula_ext_inter = (
        f"issued ~ log_rate_vol_10y_w + {spread_col} + vol_x_spread + "
        "leverage_w + cash_ratio_w + profitability_oibdp_w + tangibility_w + "
        "log_market_equity_w + term_spread_w + C(gvkey)"
    )
    res_ext_inter, reg_ext_inter = fit_firm_fe_ols(extensive_df, formula_ext_inter, cluster_col="gvkey")
    model_results.append(result_to_tidy("pivot_extensive_interaction", res_ext_inter, len(reg_ext_inter)))

# ------------------------------------
# Save outputs
# ------------------------------------
    results_df = pd.concat(model_results, ignore_index=True)
    results_path = out_dir / "regression_results_spread_pivot_tidy.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved pivot regression results to {results_path}")

    summary = {
        "pivot_intensive_baseline": {
            "nobs": int(len(reg_int_base)),
            "rsquared": float(res_int_base.rsquared),
        },
        "pivot_intensive_interaction": {
            "nobs": int(len(reg_int_inter)),
            "rsquared": float(res_int_inter.rsquared),
        },
        "pivot_extensive_baseline": {
            "nobs": int(len(reg_ext_base)),
            "rsquared": float(res_ext_base.rsquared),
        },
        "pivot_extensive_interaction": {
            "nobs": int(len(reg_ext_inter)),
            "rsquared": float(res_ext_inter.rsquared),
        },
    }
    summary_path = out_dir / "regression_results_spread_pivot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved pivot regression summary to {summary_path}")


if __name__ == "__main__":
    main()