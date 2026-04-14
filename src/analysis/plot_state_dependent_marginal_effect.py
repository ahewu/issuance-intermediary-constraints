from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from src.utils.io import load_yaml, read_parquet, ensure_dir
from src.utils.logging_utils import get_logger


def prepare_state_dependent_vars(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Preparing state-dependent variables...")
    df = df.copy().sort_values("month")

    df["credit_spread_w_lag"] = df["credit_spread_w"].shift(1)
    df["vol_x_spread_lag"] = df["log_rate_vol_10y_w"] * df["credit_spread_w_lag"]

    return df


def fit_interaction_model(df: pd.DataFrame, logger):
    logger.info("Fitting interaction model for marginal-effect plot...")

    needed = [
        "issued",
        "log_rate_vol_10y_w",
        "credit_spread_w_lag",
        "vol_x_spread_lag",
        "leverage_w",
        "cash_ratio_w",
        "profitability_oibdp_w",
        "tangibility_w",
        "log_market_equity_w",
        "term_spread_w",
        "gvkey",
    ]
    reg = df.dropna(subset=needed).copy()

    formula = """
    issued ~ log_rate_vol_10y_w
    + credit_spread_w_lag
    + vol_x_spread_lag
    + leverage_w + cash_ratio_w + profitability_oibdp_w
    + tangibility_w + log_market_equity_w + term_spread_w
    + C(gvkey)
    """

    model = smf.ols(formula, data=reg)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": reg["gvkey"]})

    logger.info(f"Interaction model fitted on {len(reg):,} rows.")
    return res, reg


def build_marginal_effect_df(res, reg: pd.DataFrame) -> pd.DataFrame:
    b1 = res.params["log_rate_vol_10y_w"]
    b3 = res.params["vol_x_spread_lag"]

    cov = res.cov_params()
    var_b1 = cov.loc["log_rate_vol_10y_w", "log_rate_vol_10y_w"]
    var_b3 = cov.loc["vol_x_spread_lag", "vol_x_spread_lag"]
    cov_b1_b3 = cov.loc["log_rate_vol_10y_w", "vol_x_spread_lag"]

    spread_min = reg["credit_spread_w_lag"].quantile(0.01)
    spread_max = reg["credit_spread_w_lag"].quantile(0.99)
    spread_grid = np.linspace(spread_min, spread_max, 200)

    effect = b1 + b3 * spread_grid
    se = np.sqrt(var_b1 + (spread_grid ** 2) * var_b3 + 2 * spread_grid * cov_b1_b3)

    out = pd.DataFrame({
        "credit_spread_w_lag": spread_grid,
        "marginal_effect": effect,
        "ci_low": effect - 1.96 * se,
        "ci_high": effect + 1.96 * se,
    })
    return out


def plot_marginal_effect(me_df: pd.DataFrame, out_path: Path, logger) -> None:
    logger.info(f"Saving marginal-effect figure to {out_path}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        me_df["credit_spread_w_lag"],
        me_df["marginal_effect"],
        linewidth=2,
        label="Marginal effect of volatility on issuance probability",
    )
    ax.fill_between(
        me_df["credit_spread_w_lag"],
        me_df["ci_low"],
        me_df["ci_high"],
        alpha=0.2,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    
    ax.set_xlabel("Lagged credit spread")
    ax.set_ylabel("Marginal effect of volatility on issuance probability")
    ax.set_title("State-dependent effect of interest rate volatility on debt issuance")

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    logger = get_logger(
        "plot_state_dependent_marginal_effect",
        log_file="logs/plot_state_dependent_marginal_effect.log",
    )
    paths = load_yaml("config/paths.yaml")

    panel_path = Path(paths["data_processed"]) / "panels" / "firm_month_regression.parquet"
    out_dir = Path(paths["data_output"]) / "figures"
    ensure_dir(out_dir)
    out_path = out_dir / "state_dependent_marginal_effect.png"

    logger.info(f"Reading panel from {panel_path}")
    df = read_parquet(panel_path)

    df = df[df["sample_regression_main"] == 1].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str).str.strip()

    df = prepare_state_dependent_vars(df, logger)
    res, reg = fit_interaction_model(df, logger)
    me_df = build_marginal_effect_df(res, reg)
    plot_marginal_effect(me_df, out_path, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()