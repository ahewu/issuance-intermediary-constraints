from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.utils.io import load_yaml, ensure_dir
from src.utils.logging_utils import get_logger


STAR_LEVELS = [
    (0.01, "***"),
    (0.05, "**"),
    (0.10, "*"),
]


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    for cutoff, s in STAR_LEVELS:
        if p < cutoff:
            return s
    return ""


def fmt_coef(coef: float, p: float) -> str:
    if pd.isna(coef):
        return ""
    return f"{coef:.4f}{stars(p)}"


def fmt_se(se: float) -> str:
    if pd.isna(se):
        return ""
    return f"({se:.4f})"


def make_two_line_entry(coef: float, se: float, p: float) -> str:
    c = fmt_coef(coef, p)
    s = fmt_se(se)
    if c == "":
        return ""
    return f"{c} \\\\ {s}"


def extract_model(df: pd.DataFrame, model_name: str, terms: list[str]) -> dict[str, str]:
    sub = df[df["model"] == model_name].copy()
    out = {}
    for term in terms:
        row = sub[sub["term"] == term]
        if len(row) == 0:
            out[term] = ""
        else:
            r = row.iloc[0]
            out[term] = make_two_line_entry(r["coef"], r["std_err"], r["p_value"])
    return out


def latex_table(
    title: str,
    label: str,
    row_labels: list[tuple[str, str]],
    col_labels: list[str],
    col_contents: list[dict[str, str]],
    footer_rows: list[tuple[str, list[str]]],
) -> str:
    ncols = len(col_labels)
    col_spec = "l" + "c" * ncols

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{title}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")

    for key, nice_label in row_labels:
        row = [nice_label]
        for content in col_contents:
            row.append(content.get(key, ""))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\midrule")

    for foot_label, vals in footer_rows:
        lines.append(foot_label + " & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\begin{minipage}{0.9\\textwidth}")
    lines.append("\\footnotesize Notes: Entries report coefficients with clustered standard errors in parentheses. "
                 "*, **, and *** denote significance at the 10\\%, 5\\%, and 1\\% levels, respectively.")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    logger = get_logger("build_regression_tables")
    paths = load_yaml("config/paths.yaml")

    results_path = Path(paths["data_processed"]) / "results" / "regression_results_state_dependent_tidy.csv"
    summary_path = Path(paths["data_processed"]) / "results" / "regression_results_state_dependent_summary.json"
    out_dir = Path(paths["data_output"]) / "tables"
    ensure_dir(out_dir)

    logger.info(f"Reading results from {results_path}")
    df = pd.read_csv(results_path)

    logger.info(f"Reading summary from {summary_path}")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    row_labels = [
        ("log_rate_vol_10y_w", "Rate volatility"),
        ("credit_spread_w_lag", "Lagged credit spread"),
        ("vol_x_spread_lag", "Volatility $\\times$ lagged spread"),
        ("leverage_w", "Leverage"),
        ("cash_ratio_w", "Cash ratio"),
        ("profitability_oibdp_w", "Profitability"),
        ("tangibility_w", "Tangibility"),
        ("log_market_equity_w", "Log market equity"),
        ("term_spread_w", "Term spread"),
    ]

    # Main table
    baseline_model = "baseline_vol_only"
    interaction_model = "interaction_main"

    baseline = extract_model(df, baseline_model, [k for k, _ in row_labels])
    interaction = extract_model(df, interaction_model, [k for k, _ in row_labels])

    main_footer = [
        ("Firm FE", ["Yes", "Yes"]),
        ("Time FE", ["Yes", "No"]),
        ("Observations", [
            f"{int(summary['baseline']['nobs']):,}",
            f"{int(summary['interaction']['nobs']):,}",
        ]),
        ("$R^2$", [
            f"{summary['baseline']['r2']:.3f}",
            f"{summary['interaction']['r2']:.3f}",
        ]),
    ]

    main_tex = latex_table(
        title="State-Dependent Effect of Volatility on Debt Issuance",
        label="tab:main_state_dependent",
        row_labels=row_labels,
        col_labels=["(1) Baseline", "(2) Interaction"],
        col_contents=[baseline, interaction],
        footer_rows=main_footer,
    )

    main_out = out_dir / "table_main_state_dependent.tex"
    main_out.write_text(main_tex)
    logger.info(f"Saved {main_out}")

    # Split-sample table
    low_model = "low_spread_sample"
    high_model = "high_spread_sample"

    split_terms = [
        ("log_rate_vol_10y_w", "Rate volatility"),
        ("leverage_w", "Leverage"),
        ("cash_ratio_w", "Cash ratio"),
        ("profitability_oibdp_w", "Profitability"),
        ("tangibility_w", "Tangibility"),
        ("log_market_equity_w", "Log market equity"),
        ("term_spread_w", "Term spread"),
    ]

    low = extract_model(df, low_model, [k for k, _ in split_terms])
    high = extract_model(df, high_model, [k for k, _ in split_terms])

    split_footer = [
        ("Firm FE", ["Yes", "Yes"]),
        ("Time FE", ["No", "No"]),
        ("Observations", [
            f"{int(summary['low_spread']['nobs']):,}",
            f"{int(summary['high_spread']['nobs']):,}",
        ]),
        ("$R^2$", [
            f"{summary['low_spread']['r2']:.3f}",
            f"{summary['high_spread']['r2']:.3f}",
        ]),
    ]

    split_tex = latex_table(
        title="Split-Sample Evidence by Credit Spread Regime",
        label="tab:split_spread",
        row_labels=split_terms,
        col_labels=["(1) Low spread", "(2) High spread"],
        col_contents=[low, high],
        footer_rows=split_footer,
    )

    split_out = out_dir / "table_split_spread.tex"
    split_out.write_text(split_tex)
    logger.info(f"Saved {split_out}")


if __name__ == "__main__":
    main()