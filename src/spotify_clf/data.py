"""Carga de datos y split train/validation."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descargá el archivo de la competencia de "
            "Kaggle y colocalo en data/raw/ (ver README.md)."
        )
    return pd.read_csv(path)


def load_train(path: Path = config.TRAIN_PATH) -> pd.DataFrame:
    return load_csv(path)


def load_test(path: Path = config.TEST_PATH) -> pd.DataFrame:
    return load_csv(path)


def get_features_and_target(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    drop_cols: list[str] | None = None,
):
    """Separa un DataFrame de train en X (features) e y (target).

    `drop_cols` son columnas que no se usan como feature (por default el id
    y las columnas de texto libre); se dropean si están presentes.
    """
    if drop_cols is None:
        drop_cols = [config.ID_COL, *config.TEXT_COLS]

    y = df[target_col]
    X = df.drop(columns=[c for c in [target_col, *drop_cols] if c in df.columns])
    return X, y


def train_val_split(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    test_size: float = 0.2,
    random_state: int = config.RANDOM_STATE,
):
    """Split estratificado por género, para que la validación local tenga
    la misma proporción de clases que el train.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
