import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.io import load_yaml, ensure_dir, save_parquet
from src.utils.logging_utils import get_logger
from src.utils.checks import assert_unique


def main():
    print("Starting smoke test...")
    logger = get_logger("smoke_test")

    paths = load_yaml("config/paths.yaml")
    print("Loaded YAML")

    ensure_dir(paths["data_output"])
    print("Ensured output dir")

    logger.info("Loaded config successfully.")

    df = pd.DataFrame(
        {
            "gvkey": [1001, 1002, 1003],
            "month": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-01-31"]),
            "test_value": [1.0, 2.0, 3.0],
        }
    )

    print("Created DataFrame")
    assert_unique(df, ["gvkey", "month"])
    print("Passed uniqueness check")

    outpath = Path(paths["data_output"]) / "smoke_test.parquet"
    save_parquet(df, outpath)
    print(f"Wrote file to {outpath}")

    logger.info(f"Smoke test complete. Wrote file to {outpath}")


if __name__ == "__main__":
    main()
