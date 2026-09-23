"""Búsqueda de hiperparámetros optimizando la métrica oficial (F1-Macro)."""
from typing import Literal

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from . import config
from .metrics import f1_macro_scorer
from .models import ModelSpec


def tune_model(
    spec: ModelSpec,
    X,
    y,
    search_type: Literal["grid", "random"] = "random",
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring=f1_macro_scorer,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """Corre GridSearchCV o RandomizedSearchCV sobre `spec.pipeline`.

    Usa StratifiedKFold porque las clases (géneros) están relativamente
    balanceadas pero conviene preservar la proporción en cada fold, en
    especial si se experimenta con submuestreos del dataset.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if search_type == "grid":
        search = GridSearchCV(
            estimator=spec.pipeline,
            param_grid=spec.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=spec.pipeline,
            param_distributions=spec.param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            refit=True,
        )
    else:
        raise ValueError(f"search_type debe ser 'grid' o 'random', recibido: {search_type}")

    search.fit(X, y)
    return search
