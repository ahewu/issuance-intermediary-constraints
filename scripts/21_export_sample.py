import pandas as pd
from pathlib import Path

path = Path("data/processed/panels/firm_month_regression.parquet")

df = pd.read_parquet(path)

# Ensure proper ordering
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.sort_values("month")

# Create lag
if "credit_spread_w" in df.columns:
    df["credit_spread_w_lag"] = df["credit_spread_w"].shift(1)

cols_needed = [
    "issued",
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

# Keep only columns that exist
cols_needed = [c for c in cols_needed if c in df.columns]

df_clean = (
    df[cols_needed]
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
)

sample = df_clean.head(50)

out_path = Path("data/sample_data.csv")
sample.to_csv(out_path, index=False)

print(f"Saved sample to {out_path}")