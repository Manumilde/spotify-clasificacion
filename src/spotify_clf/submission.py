"""Generación del CSV de submission para Kaggle.

La métrica oficial de la competencia es F1-Macro sobre la predicción top-1
(no top-3), así que la submission lleva una sola etiqueta por fila.
"""
from pathlib import Path

import pandas as pd

from . import config


def build_submission(
    ids: pd.Series,
    predictions,
    id_col: str = config.ID_COL,
    target_col: str = config.TARGET_COL,
) -> pd.DataFrame:
    return pd.DataFrame({id_col: ids.values, target_col: predictions})


def save_submission(
    submission_df: pd.DataFrame,
    path: Path = config.SUBMISSIONS_DIR / "submission.csv",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(path, index=False)
    return path
