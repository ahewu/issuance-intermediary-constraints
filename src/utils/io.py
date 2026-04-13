from pathlib import Path
from typing import Union
import pandas as pd
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML file is empty: {path}")
    return data


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_parquet(path):
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)