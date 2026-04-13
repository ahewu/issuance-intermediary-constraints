import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import logging
from pathlib import Path


# Setup
def get_logger():
    logger = logging.getLogger("run_regressions_state_dependent")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Helper functions
def prepare_data(df, logger):
    logger.info("Preparing regression data...")

    # Lag spread (by month)
    df = df.sort_values(["month"])
    df["credit_spread_w_lag"] = df["credit_spread_w"].shift(1)

    # Drop first month (missing lag)
    df = df.dropna(subset=["credit_spread_w_lag"])

    # Create interaction
    df["vol_x_spread_lag"] = df["log_rate_vol_10y_w"] * df["credit_spread_w_lag"]

    # Create high/low spread split
    median_spread = df["credit_spread_w_lag"].median()
    df["high_spread"] = (df["credit_spread_w_lag"] >= median_spread).astype(int)

    logger.info(f"Median lagged spread: {median_spread:.4f}")

    return df


def run_fe_ols(df, formula, cluster_col="gvkey"):
    model = smf.ols(formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return res


# Main regressions
def main():

    logger = get_logger()

    # Paths
    input_path = Path("data/processed/panels/firm_month_regression.parquet")
    output_dir = Path("data/processed/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading regression-ready panel from {input_path}")
    df = pd.read_parquet(input_path)

    # Prepare data
    df = prepare_data(df, logger)

    # Keep main regression sample
    df_main = df.dropna(subset=[
        "issued",
        "log_rate_vol_10y_w",
        "credit_spread_w_lag",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "term_spread_w"
    ])

    logger.info(f"Final regression sample rows: {len(df_main):,}")
    logger.info(f"Unique firms: {df_main['gvkey'].nunique():,}")


    # 1. Baseline (no spread)
    logger.info("Running baseline (real-options only)")

    formula_baseline = """
    issued ~ log_rate_vol_10y_w
    + leverage_w + cash_ratio_w + profitability_oibdp_w
    + tangibility_w + log_market_equity_w + term_spread_w
    + C(gvkey) + C(month)
    """

    res_baseline = run_fe_ols(df_main, formula_baseline)


    # 2. Interaction (MAIN SPEC)
    logger.info("Running state-dependent interaction regression")

    formula_interaction = """
    issued ~ log_rate_vol_10y_w
    + credit_spread_w_lag
    + vol_x_spread_lag
    + leverage_w + cash_ratio_w + profitability_oibdp_w
    + tangibility_w + log_market_equity_w + term_spread_w
    + C(gvkey)
    """

    res_interaction = run_fe_ols(df_main, formula_interaction)


    # 3. Split-sample test
    logger.info("Running split-sample regressions")

    df_low = df_main[df_main["high_spread"] == 0]
    df_high = df_main[df_main["high_spread"] == 1]

    def clean_for_reg(df_in, cols):
        out = df_in.copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.dropna(subset=cols)

    needed_split = ["issued", "log_rate_vol_10y_w", "log_market_equity_w", "term_spread_w", "gvkey"]
    df_low = clean_for_reg(df_low, needed_split)
    df_high = clean_for_reg(df_high, needed_split)

    formula_split = """
    issued ~ log_rate_vol_10y_w
    + leverage_w + cash_ratio_w + profitability_oibdp_w
    + tangibility_w + log_market_equity_w + term_spread_w
    + C(gvkey)
    """

# Drop firms with no variation in issued within each split sample
    def keep_firms_with_variation(df_in):
        var = df_in.groupby("gvkey")["issued"].nunique()
        good = var[var > 1].index
        return df_in[df_in["gvkey"].isin(good)].copy()

    df_low = keep_firms_with_variation(df_low)
    df_high = keep_firms_with_variation(df_high)

    res_low = run_fe_ols(df_low, formula_split)
    res_high = run_fe_ols(df_high, formula_split)


    # Save results
    def tidy_results(res, name):
        df_out = pd.DataFrame({
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t_stat": res.tvalues.values,
            "p_value": res.pvalues.values,
            "model": name,
            "nobs": res.nobs
        })
        return df_out

    results = pd.concat([
        tidy_results(res_baseline, "baseline_vol_only"),
        tidy_results(res_interaction, "interaction_main"),
        tidy_results(res_low, "low_spread_sample"),
        tidy_results(res_high, "high_spread_sample"),
    ])

    results_path = output_dir / "regression_results_state_dependent_tidy.csv"
    results.to_csv(results_path, index=False)

    summary = {
        "baseline": {"nobs": res_baseline.nobs, "r2": res_baseline.rsquared},
        "interaction": {"nobs": res_interaction.nobs, "r2": res_interaction.rsquared},
        "low_spread": {"nobs": res_low.nobs, "r2": res_low.rsquared},
        "high_spread": {"nobs": res_high.nobs, "r2": res_high.rsquared},
    }

    summary_path = output_dir / "regression_results_state_dependent_summary.json"
    pd.Series(summary).to_json(summary_path)

    logger.info(f"Saved results to {results_path}")
    logger.info(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()