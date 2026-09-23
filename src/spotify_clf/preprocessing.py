"""Preprocesamiento de features.

Decisión de diseño (ver informe/notebook para la justificación completa):
- Features numéricas continuas -> StandardScaler (los modelos lineales y
  basados en distancia lo necesitan; a los de árboles no los perjudica).
- Features categóricas de baja cardinalidad (key, mode, time_signature,
  explicit) -> OneHotEncoder, porque son códigos sin orden real (o con muy
  pocos niveles) y no una magnitud continua.
- Columnas de texto libre (artists, album_name, track_name) e identificador
  (track_id) -> se descartan del pipeline de features "out of the box".
  Quedan disponibles en config.TEXT_COLS por si se quiere experimentar con
  feature engineering de texto (p. ej. longitud de nombre, cantidad de
  artistas) como extensión.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(
    numeric_cols: list[str] = config.NUMERIC_COLS,
    categorical_cols: list[str] = config.CATEGORICAL_COLS,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
