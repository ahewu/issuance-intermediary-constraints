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


def preview_link_tables(db, logger) -> None:
    logger.info("Previewing WRDS bond/CRSP/Compustat link tables...")

    try:
        bond_link = db.get_table(
            library="wrdsapps_link_crsp_bond",
            table="bondcrsp_link",
            obs=5,
        )
        logger.info(f"Preview of wrdsapps_link_crsp_bond.bondcrsp_link:\n{bond_link.head()}")
        logger.info(f"Columns in bondcrsp_link: {list(bond_link.columns)}")
    except Exception as e:
        logger.warning(f"Could not preview bondcrsp_link: {e}")

    try:
        ccm = db.get_table(
            library="crsp",
            table="ccmxpf_linktable",
            obs=5,
        )
        logger.info(f"Preview of crsp.ccmxpf_linktable:\n{ccm.head()}")
        logger.info(f"Columns in ccmxpf_linktable: {list(ccm.columns)}")
    except Exception as e:
        logger.warning(f"Could not preview ccmxpf_linktable: {e}")


def build_compustat_link_base(comp: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Building Compustat fallback link base...")

    comp = comp.copy()
    comp["cusip"] = comp["cusip"].astype("string").str.strip()
    comp["comp_cusip6"] = comp["cusip"].str[:6]
    comp["comp_name_std"] = standardize_name(comp["conm"])
    comp["tic"] = comp["tic"].astype("string").str.strip().str.upper()

    keep_cols = ["gvkey", "comp_cusip6", "comp_name_std", "tic", "datadate"]
    keep_cols = [c for c in keep_cols if c in comp.columns]
    comp = comp[keep_cols].copy()

    comp = comp.dropna(subset=["gvkey"])
    comp = comp.sort_values(["gvkey", "datadate"])
    comp = comp.drop_duplicates(subset=["gvkey", "comp_cusip6"], keep="last")

    logger.info(f"Fallback Compustat link base rows: {len(comp):,}")
    return comp


def load_bond_crsp_link(db, logger) -> pd.DataFrame:
    """
    Adjust this function if your preview shows different column names.
    Most likely needed fields:
    - cusip or cusip_id
    - permno
    """
    logger.info("Loading WRDS bond->CRSP link table...")

    # Common likely candidates. Start broad with get_table then trim.
    df = db.get_table(
        library="wrdsapps_link_crsp_bond",
        table="bondcrsp_link",
    )

    logger.info(f"Loaded bondcrsp_link rows: {len(df):,}")
    logger.info(f"bondcrsp_link columns: {list(df.columns)}")

    # Try to detect likely columns
    cols_lower = {c.lower(): c for c in df.columns}

    cusip_col = None
    permno_col = None

    for candidate in ["cusip_id", "cusip", "complete_cusip"]:
        if candidate in cols_lower:
            cusip_col = cols_lower[candidate]
            break

    for candidate in ["permno", "lpermno"]:
        if candidate in cols_lower:
            permno_col = cols_lower[candidate]
            break

    if cusip_col is None or permno_col is None:
        raise ValueError(
            "Could not identify cusip/permno columns in bondcrsp_link. "
            "Check previewed column names in the log."
        )

    out = df[[cusip_col, permno_col]].copy()
    out = out.rename(columns={cusip_col: "complete_cusip", permno_col: "permno"})
    out["complete_cusip"] = out["complete_cusip"].astype("string").str.strip()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    out = out.dropna(subset=["complete_cusip", "permno"]).drop_duplicates()

    logger.info(f"Usable bond->CRSP links: {len(out):,}")
    return out


def load_crsp_compustat_link(db, logger) -> pd.DataFrame:
    """
    Standard CRSP/Compustat link table.
    We keep only reasonable link types and primaries.
    """
    logger.info("Loading CRSP->Compustat link table...")

    df = db.get_table(
        library="crsp",
        table="ccmxpf_linktable",
    )

    logger.info(f"Loaded ccmxpf_linktable rows: {len(df):,}")
    logger.info(f"ccmxpf_linktable columns: {list(df.columns)}")

    # Standard columns expected
    needed = ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected ccmxpf_linktable columns: {missing}")

    out = df[needed].copy()
    out = out.rename(columns={"lpermno": "permno"})
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    out["gvkey"] = out["gvkey"].astype("string").str.strip()
    out["linkdt"] = pd.to_datetime(out["linkdt"], errors="coerce")
    out["linkenddt"] = pd.to_datetime(out["linkenddt"], errors="coerce")

    # Keep standard primary link types
    out = out[
        out["linktype"].astype("string").isin(["LC", "LU", "LS"])
        & out["linkprim"].astype("string").isin(["P", "C"])
    ].copy()

    out = out.dropna(subset=["gvkey", "permno"]).drop_duplicates()

    logger.info(f"Usable CRSP->Compustat links: {len(out):,}")
    return out


def link_via_crsp(
    fisd: pd.DataFrame,
    bond_crsp: pd.DataFrame,
    ccm: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    logger.info("Linking FISD -> CRSP -> Compustat...")

    fisd = fisd.copy()
    fisd["complete_cusip"] = fisd["complete_cusip"].astype("string").str.strip()
    fisd["issue_date"] = pd.to_datetime(fisd["issue_date"], errors="coerce")
    fisd["fisd_name_std"] = standardize_name(fisd["prospectus_issuer_name"])

    merged = fisd.merge(
        bond_crsp,
        on="complete_cusip",
        how="left",
    )

    logger.info(f"Rows with permno after bond->CRSP link: {merged['permno'].notna().sum():,}")

    merged = merged.merge(
        ccm,
        on="permno",
        how="left",
        suffixes=("", "_ccm"),
    )

    # Enforce link-date validity if available
    valid_date = (
        merged["gvkey"].notna()
        & (
            merged["linkdt"].isna()
            | (merged["issue_date"] >= merged["linkdt"])
        )
        & (
            merged["linkenddt"].isna()
            | (merged["issue_date"] <= merged["linkenddt"])
        )
    )

    merged.loc[~valid_date, "gvkey"] = pd.NA

    merged["link_method"] = pd.NA
    merged.loc[merged["gvkey"].notna(), "link_method"] = "crsp_ccm"

    logger.info(f"Rows with gvkey after CRSP/CCM link: {merged['gvkey'].notna().sum():,}")

    return merged


def link_by_cusip6_fallback(
    fisd: pd.DataFrame,
    comp_link: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    logger.info("Applying fallback FISD->Compustat cusip6 link...")

    fisd = fisd.copy()
    fisd["fisd_cusip6"] = fisd["complete_cusip"].astype("string").str[:6]
    fisd["fisd_name_std"] = standardize_name(fisd["prospectus_issuer_name"])

    merged = fisd.merge(
        comp_link,
        left_on="fisd_cusip6",
        right_on="comp_cusip6",
        how="left",
        suffixes=("", "_comp"),
    )

    merged["fallback_match"] = merged["gvkey"].notna()

    logger.info(f"Rows with gvkey after fallback merge: {merged['gvkey'].notna().sum():,}")
    return merged


def resolve_fallback_ambiguity(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Resolving fallback ambiguity...")

    df = df.copy()
    issue_key = "issue_id" if "issue_id" in df.columns else "complete_cusip"

    candidate_counts = (
        df.groupby(issue_key)["gvkey"]
        .nunique(dropna=True)
        .rename("n_gvkey_candidates")
        .reset_index()
    )
    df = df.merge(candidate_counts, on=issue_key, how="left")

    unique_match = df["n_gvkey_candidates"].fillna(0).eq(1)
    exact_name_match = (
        df["gvkey"].notna()
        & (df["fisd_name_std"] == df["comp_name_std"])
    )

    preferred = df[unique_match | exact_name_match].copy()
    preferred = preferred.sort_values([issue_key, "gvkey", "datadate"], na_position="last")
    preferred = preferred.drop_duplicates(subset=[issue_key], keep="first")

    all_issues = df[[issue_key]].drop_duplicates()
    preferred = all_issues.merge(preferred, on=issue_key, how="left")
    preferred["link_method"] = "cusip6_fallback"

    return preferred


def combine_primary_and_fallback(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    logger.info("Combining primary CRSP-based and fallback links...")

    issue_key = "issue_id" if "issue_id" in primary.columns else "complete_cusip"

    primary_small = primary[[issue_key, "gvkey", "link_method"]].copy()
    primary_small = primary_small.drop_duplicates(subset=[issue_key], keep="first")

    fallback_small = fallback[[issue_key, "gvkey", "link_method", "n_gvkey_candidates"]].copy()
    fallback_small = fallback_small.drop_duplicates(subset=[issue_key], keep="first")

    combined = primary_small.merge(
        fallback_small,
        on=issue_key,
        how="outer",
        suffixes=("_primary", "_fallback"),
    )

    combined["gvkey_final"] = combined["gvkey_primary"]
    combined["link_method_final"] = combined["link_method_primary"]

    use_fallback = combined["gvkey_final"].isna() & combined["gvkey_fallback"].notna()
    combined.loc[use_fallback, "gvkey_final"] = combined.loc[use_fallback, "gvkey_fallback"]
    combined.loc[use_fallback, "link_method_final"] = "cusip6_fallback"

    combined["link_status"] = "unmatched"
    combined.loc[combined["gvkey_final"].notna(), "link_status"] = "matched"

    logger.info(f"Combined matched issues: {combined['gvkey_final'].notna().sum():,}")
    logger.info(f"Combined unmatched issues: {combined['gvkey_final'].isna().sum():,}")

    return combined


def propagate_gvkey_within_issuer(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Applying issuer-level gvkey propagation...")

    df = df.copy()

    if "issuer_id" not in df.columns:
        logger.warning("issuer_id not found; skipping issuer-level propagation.")
        return df

    # Build issuer -> modal gvkey map using already matched issues only
    matched = df[df["gvkey"].notna()].copy()

    issuer_map = (
        matched.groupby("issuer_id")["gvkey"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        .rename("gvkey_issuer")
        .reset_index()
    )

    # Optional diagnostic: how many distinct gvkeys per issuer among matched issues
    issuer_n_gvkey = (
        matched.groupby("issuer_id")["gvkey"]
        .nunique()
        .rename("issuer_n_gvkey")
        .reset_index()
    )

    issuer_map = issuer_map.merge(issuer_n_gvkey, on="issuer_id", how="left")

    df = df.merge(issuer_map, on="issuer_id", how="left")

    # Only propagate when issuer has a unique matched gvkey
    mask = (
        df["gvkey"].isna()
        & df["gvkey_issuer"].notna()
        & df["issuer_n_gvkey"].eq(1)
    )

    propagated_count = int(mask.sum())

    df.loc[mask, "gvkey"] = df.loc[mask, "gvkey_issuer"]
    df.loc[mask, "link_method"] = "issuer_propagation"
    df.loc[mask, "link_status"] = "matched"

    logger.info(f"Issues filled via issuer propagation: {propagated_count:,}")

    # Cleanup helper columns
    df = df.drop(columns=["gvkey_issuer", "issuer_n_gvkey"], errors="ignore")

    logger.info(f"Link rate after issuer propagation: {df['gvkey'].notna().mean():.2%}")

    return df


def main() -> None:
    logger = get_logger("link_fisd_compustat_crsp", log_file="logs/link_fisd_compustat_crsp.log")
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

    import wrds
    db = wrds.Connection()

    preview_link_tables(db, logger)

    bond_crsp = load_bond_crsp_link(db, logger)
    ccm = load_crsp_compustat_link(db, logger)

    primary = link_via_crsp(fisd, bond_crsp, ccm, logger)

    comp_link = build_compustat_link_base(comp, logger)
    fallback_raw = link_by_cusip6_fallback(fisd, comp_link, logger)
    fallback = resolve_fallback_ambiguity(fallback_raw, logger)

    combined = combine_primary_and_fallback(primary, fallback, logger)

    issue_key = "issue_id" if "issue_id" in fisd.columns else "complete_cusip"

    link_table = fisd[[issue_key, "complete_cusip", "prospectus_issuer_name"]].copy()
    link_table = link_table.merge(
        combined[[issue_key, "gvkey_final", "link_method_final", "link_status"]],
        on=issue_key,
        how="left",
    )
    link_table = link_table.rename(
        columns={
            "gvkey_final": "gvkey",
            "link_method_final": "link_method",
        }
    )

    link_table_path = links_dir / "fisd_compustat_link_table.parquet"
    save_parquet(link_table, link_table_path)
    logger.info(f"Saved link table to {link_table_path}")

    fisd_linked = fisd.merge(
        link_table[[issue_key, "gvkey", "link_method", "link_status"]],
        on=issue_key,
        how="left",
    )

    fisd_linked = propagate_gvkey_within_issuer(fisd_linked, logger)

    fisd_linked_path = fisd_out_dir / "fisd_issues_linked.parquet"
    save_parquet(fisd_linked, fisd_linked_path)
    logger.info(f"Saved linked FISD to {fisd_linked_path}")

    logger.info(f"Final FISD linked rows: {len(fisd_linked):,}")
    logger.info(f"Rows with gvkey: {fisd_linked['gvkey'].notna().sum():,}")
    logger.info(
        f"Link rate overall: {fisd_linked['gvkey'].notna().mean():.2%}"
    )
    logger.info(
        f"Link rate for 144A: {fisd_linked.loc[fisd_linked['is_144a'], 'gvkey'].notna().mean():.2%}"
    )
    logger.info(
        f"Link rate for non-144A: {fisd_linked.loc[~fisd_linked['is_144a'], 'gvkey'].notna().mean():.2%}"
    )
    logger.info(
        f"Link rate for private placements: {fisd_linked.loc[fisd_linked['is_private_placement'], 'gvkey'].notna().mean():.2%}"
    )
    logger.info(
        f"Median amount linked: {fisd_linked.loc[fisd_linked['gvkey'].notna(), 'amount_issued'].median():,.0f}"
    )
    logger.info(
        f"Median amount unmatched: {fisd_linked.loc[fisd_linked['gvkey'].isna(), 'amount_issued'].median():,.0f}"
    )
    logger.info(
        f"Link method counts: {fisd_linked['link_method'].astype('string').value_counts(dropna=False).to_dict()}"
    )
    logger.info(
        f"Issuers with multiple gvkeys (matched set): "
        f"{(fisd_linked.groupby('issuer_id')['gvkey'].nunique() > 1).sum()}"
    )

if __name__ == "__main__":
    main()