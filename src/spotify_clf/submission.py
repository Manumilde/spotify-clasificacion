"""Generación del CSV de submission para Kaggle.

La métrica oficial de la competencia es F1-Macro sobre la predicción top-1
(no top-3), así que la submission lleva una sola etiqueta por fila. El
formato exigido por esta competencia es exactamente dos columnas,
`Id,Expected` (ver consigna) — coincide con `config.ID_COL`/
`config.TARGET_COL` tal cual vienen en el dataset.
"""
from pathlib import Path

import pandas as pd

from . import config


def build_submission(
    ids: pd.Series,
    predictions,
    id_col: str = config.ID_COL,
    target_col: str = config.SUBMISSION_TARGET_COL,
) -> pd.DataFrame:
    return pd.DataFrame({id_col: ids.values, target_col: predictions})


def save_submission(
    submission_df: pd.DataFrame,
    path: Path = config.SUBMISSIONS_DIR / "submission.csv",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(path, index=False)
    return path


def validate_submission_format(
    submission_df: pd.DataFrame,
    sample_path: Path = config.SAMPLE_SUBMISSION_PATH,
    id_col: str = config.ID_COL,
    target_col: str = config.SUBMISSION_TARGET_COL,
) -> None:
    """Chequea la submission contra `sample_submission.csv` antes de subirla
    a Kaggle: mismas columnas (y orden), mismo set de `Id` y sin nulos.
    Si `sample_path` no existe, no hace nada (no todas las competencias lo
    proveen). Lanza `AssertionError` con un mensaje descriptivo si algo no
    coincide.
    """
    if not Path(sample_path).exists():
        return

    sample_df = pd.read_csv(sample_path)

    assert list(submission_df.columns) == list(sample_df.columns), (
        f"Columnas esperadas {list(sample_df.columns)}, "
        f"encontradas {list(submission_df.columns)}"
    )
    assert len(submission_df) == len(sample_df), (
        f"Se esperaban {len(sample_df)} filas, hay {len(submission_df)}"
    )
    assert set(submission_df[id_col]) == set(sample_df[id_col]), (
        "El set de Id de la submission no coincide con el de sample_submission.csv"
    )
    assert submission_df[target_col].notna().all(), (
        f"Hay valores nulos en la columna '{target_col}'"
    )
