from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.io import load_yaml, ensure_dir, read_parquet, save_parquet
from src.utils.logging_utils import get_logger


def standardize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_compustat_link_base(comp: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Building Compustat link base...")

    comp = comp.copy()

    comp["cusip"] = comp["cusip"].astype("string").str.strip()
    comp["comp_cusip6"] = comp["cusip"].str[:6]
    comp["comp_name_std"] = standardize_name(comp["conm"])
    comp["tic"] = comp["tic"].astype("string").str.strip().str.upper()

    # Keep one row per gvkey-cusip6 to avoid exploding merges
    keep_cols = ["gvkey", "comp_cusip6", "comp_name_std", "tic", "datadate"]
    keep_cols = [c for c in keep_cols if c in comp.columns]
    comp = comp[keep_cols].copy()

    comp = comp.dropna(subset=["gvkey"])
    comp = comp.sort_values(["gvkey", "datadate"])

    # For each gvkey-cusip6 keep latest record
    comp = comp.drop_duplicates(subset=["gvkey", "comp_cusip6"], keep="last")

    logger.info(f"Compustat link base rows: {len(comp):,}")
    logger.info(f"Unique gvkeys in link base: {comp['gvkey'].nunique():,}")

    return comp


def link_by_cusip6(fisd: pd.DataFrame, comp_link: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Linking FISD to Compustat by cusip6...")

    fisd = fisd.copy()
    fisd["complete_cusip"] = fisd["complete_cusip"].astype("string").str.strip()
    fisd["fisd_cusip6"] = fisd["complete_cusip"].str[:6]
    fisd["fisd_name_std"] = standardize_name(fisd["prospectus_issuer_name"])

    merged = fisd.merge(
        comp_link,
        left_on="fisd_cusip6",
        right_on="comp_cusip6",
        how="left",
        suffixes=("", "_comp"),
    )

    merged["link_method"] = pd.NA
    merged.loc[merged["gvkey"].notna(), "link_method"] = "cusip6"

    logger.info(f"Rows after cusip6 merge: {len(merged):,}")
    logger.info(f"Rows with non-null gvkey after cusip6 merge: {merged['gvkey'].notna().sum():,}")

    return merged


def resolve_ambiguous_links(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Resolving ambiguous gvkey matches...")

    df = df.copy()

    # Count candidate gvkeys per issue
    issue_key = "issue_id" if "issue_id" in df.columns else "complete_cusip"
    candidate_counts = (
        df.groupby(issue_key)["gvkey"]
        .nunique(dropna=True)
        .rename("n_gvkey_candidates")
        .reset_index()
    )
    df = df.merge(candidate_counts, on=issue_key, how="left")

    # Case 1: unique gvkey candidate -> keep
    unique_match = df["n_gvkey_candidates"].fillna(0).eq(1)

    # Case 2: multiple gvkeys -> try exact standardized name match
    exact_name_match = (
        df["gvkey"].notna()
        & (df["fisd_name_std"] == df["comp_name_std"])
    )

    # Keep exact-name matches where available
    preferred = df[unique_match | exact_name_match].copy()

    # If still duplicated at issue level, keep first deterministically
    preferred = preferred.sort_values(
        [issue_key, "gvkey", "datadate"],
        na_position="last"
    )
    preferred = preferred.drop_duplicates(subset=[issue_key], keep="first")

    # Bring back unmatched issues
    all_issues = df[[issue_key]].drop_duplicates()
    preferred = all_issues.merge(preferred, on=issue_key, how="left")

    preferred["link_status"] = "unmatched"
    preferred.loc[preferred["gvkey"].notna(), "link_status"] = "matched"

    logger.info(f"Final linked issues: {len(preferred):,}")
    logger.info(f"Matched issues: {preferred['gvkey'].notna().sum():,}")
    logger.info(f"Unmatched issues: {preferred['gvkey'].isna().sum():,}")

    return preferred


def main() -> None:
    logger = get_logger("link_fisd_compustat", log_file="logs/link_fisd_compustat.log")
    paths = load_yaml("config/paths.yaml")

    fisd_path = Path(paths["data_processed"]) / "fisd" / "fisd_issues_clean.parquet"
    comp_path = Path(paths["data_processed"]) / "compustat" / "compustat_q_clean.parquet"

    links_dir = Path(paths["data_processed"]) / "links"
    fisd_out_dir = Path(paths["data_processed"]) / "fisd"

    ensure_dir(links_dir)
    ensure_dir(fisd_out_dir)

    logger.info(f"Reading cleaned FISD from {fisd_path}")
    fisd = read_parquet(fisd_path)

    logger.info(f"Reading cleaned Compustat from {comp_path}")
    comp = read_parquet(comp_path)

    comp_link = build_compustat_link_base(comp, logger)
    merged = link_by_cusip6(fisd, comp_link, logger)
    linked = resolve_ambiguous_links(merged, logger)

    link_table_cols = [
        "issue_id",
        "complete_cusip",
        "fisd_cusip6",
        "prospectus_issuer_name",
        "gvkey",
        "tic",
        "comp_name_std",
        "link_method",
        "link_status",
        "n_gvkey_candidates",
    ]
    link_table_cols = [c for c in link_table_cols if c in linked.columns]

    link_table = linked[link_table_cols].copy()
    link_table_path = links_dir / "fisd_compustat_link_table.parquet"
    save_parquet(link_table, link_table_path)
    logger.info(f"Saved link table to {link_table_path}")

    # Merge final gvkey back onto cleaned FISD
    issue_key = "issue_id" if "issue_id" in fisd.columns else "complete_cusip"
    final_link_cols = [issue_key, "gvkey", "link_method", "link_status"]
    final_link = linked[final_link_cols].copy()

    fisd_linked = fisd.merge(final_link, on=issue_key, how="left")
    fisd_linked_path = fisd_out_dir / "fisd_issues_linked.parquet"
    save_parquet(fisd_linked, fisd_linked_path)
    logger.info(f"Saved linked FISD to {fisd_linked_path}")

    logger.info(f"Final FISD linked rows: {len(fisd_linked):,}")
    logger.info(f"Rows with gvkey: {fisd_linked['gvkey'].notna().sum():,}")
    logger.info(f"Link rate: {fisd_linked['gvkey'].notna().mean():.2%}")


if __name__ == "__main__":
    main()