"""Registro de modelos y sus grillas de hiperparámetros para la búsqueda.

Cada entrada de MODEL_REGISTRY arma un Pipeline (preprocesamiento +
clasificador) y una grilla de hiperparámetros pensada para
GridSearchCV/RandomizedSearchCV. Agregar un modelo nuevo es agregar una
entrada acá; el resto del código (tuning.py, evaluate.py, scripts/) es
agnóstico al modelo concreto.
"""
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor

try:
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - xgboost es opcional
    _HAS_XGBOOST = False


if _HAS_XGBOOST:

    class LabelEncodedXGBClassifier(XGBClassifier):
        """XGBClassifier (a partir de xgboost >= 2.x) ya no acepta
        etiquetas de texto: exige que `y` venga codificado como enteros
        0..n_clases-1, a diferencia del resto de los clasificadores de
        sklearn que se usan en este proyecto (que aceptan strings
        directamente). Este wrapper codifica/decodifica las etiquetas de
        forma transparente para poder usarlo dentro del mismo Pipeline,
        con la misma interfaz (fit/predict/predict_proba) que los demás
        modelos del registro.
        """

        def fit(self, X, y, **kwargs):
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            # Mientras corre super().fit(), XGBoost valida internamente
            # que self.classes_ coincida con np.unique(y) del propio y
            # que se le pasa (enteros acá) — con self._fitting=True nuestra
            # property cae al comportamiento nativo (np.arange(n_classes_))
            # para no romper esa validación interna.
            self._fitting = True
            try:
                super().fit(X, y_encoded, **kwargs)
            finally:
                self._fitting = False
            return self

        @property
        def classes_(self):
            # `classes_` es una property de solo lectura en XGBClassifier.
            # Fuera de fit(), la pisamos para exponer las etiquetas
            # originales en vez de los enteros 0..n-1: predict_proba ya
            # devuelve las columnas en ese mismo orden (alfabético), así
            # que no hace falta reordenar nada.
            if getattr(self, "_fitting", False) or not hasattr(self, "_label_encoder"):
                return super().classes_
            return self._label_encoder.classes_

        def predict(self, X):
            y_encoded = super().predict(X)
            return self._label_encoder.inverse_transform(y_encoded)


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: dict[str, Any]


def _make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def get_model_registry(random_state: int = config.RANDOM_STATE) -> dict[str, ModelSpec]:
    registry: dict[str, ModelSpec] = {}

    # --- Baseline lineal ---------------------------------------------
    registry["logistic_regression"] = ModelSpec(
        name="logistic_regression",
        pipeline=_make_pipeline(
            LogisticRegression(max_iter=2000, random_state=random_state)
        ),
        param_grid={
            "classifier__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            "classifier__class_weight": [None, "balanced"],
        },
    )

    # --- Ensamble de árboles (bagging) --------------------------------
    registry["random_forest"] = ModelSpec(
        name="random_forest",
        pipeline=_make_pipeline(
            RandomForestClassifier(random_state=random_state, n_jobs=-1)
        ),
        param_grid={
            "classifier__n_estimators": [200, 400, 600, 800],
            "classifier__max_depth": [None, 10, 20, 30, 40],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier__class_weight": [None, "balanced"],
        },
    )

    # --- Boosting (nativo de sklearn, sin dependencias extra) ---------
    registry["hist_gradient_boosting"] = ModelSpec(
        name="hist_gradient_boosting",
        pipeline=_make_pipeline(
            HistGradientBoostingClassifier(random_state=random_state)
        ),
        param_grid={
            "classifier__learning_rate": [0.01, 0.03, 0.1, 0.3],
            "classifier__max_leaf_nodes": [15, 31, 63, 127],
            "classifier__max_iter": [100, 200, 300, 500],
            "classifier__l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "classifier__min_samples_leaf": [10, 20, 30],
        },
    )

    # --- Basado en distancia (útil para contrastar con los anteriores) -
    registry["knn"] = ModelSpec(
        name="knn",
        pipeline=_make_pipeline(KNeighborsClassifier()),
        param_grid={
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],  # 1 = distancia Manhattan, 2 = Euclídea
        },
    )

    if _HAS_XGBOOST:
        registry["xgboost"] = ModelSpec(
            name="xgboost",
            pipeline=_make_pipeline(
                LabelEncodedXGBClassifier(
                    random_state=random_state,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                )
            ),
            param_grid={
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [4, 6, 8],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__subsample": [0.8, 1.0],
            },
        )

    return registry
