import pandas as pd


def assert_unique(df: pd.DataFrame, cols: list[str]) -> None:
    dupes = df.duplicated(subset=cols).sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate rows for key {cols}")


def missing_summary(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)