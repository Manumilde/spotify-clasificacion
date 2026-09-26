"""Model selection: comparación de modelos con F1-Macro y MAP@3."""
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config
from .metrics import f1_macro_scorer, map_at_3_scorer


@dataclass
class EvalResult:
    model_name: str
    f1_macro_mean: float
    f1_macro_std: float
    map_at_3_mean: float
    map_at_3_std: float
    fold_scores: dict = field(default_factory=dict)


def cross_validate_model(
    name: str,
    pipeline,
    X,
    y,
    cv_folds: int = 5,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
) -> EvalResult:
    """Evalúa un pipeline ya instanciado (con hiperparámetros fijos, p. ej.
    el `best_estimator_` de una búsqueda) con F1-Macro y MAP@3 en la misma
    partición de cross-validation, para poder comparar ambas métricas
    modelo a modelo.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {"f1_macro": f1_macro_scorer, "map_at_3": map_at_3_scorer}
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=False
    )

    if pd.isna(scores["test_f1_macro"]).any() or pd.isna(scores["test_map_at_3"]).any():
        raise RuntimeError(
            f"cross_validate_model('{name}') dio NaN en algún fold. Esto suele "
            "indicar que un scorer está fallando dentro de cross_validate (sklearn "
            "lo tapa como NaN en vez de propagar el error) — no confíes en esta "
            "comparación de modelos hasta resolverlo. Probá n_jobs=1 para ver el "
            "traceback real."
        )

    return EvalResult(
        model_name=name,
        f1_macro_mean=scores["test_f1_macro"].mean(),
        f1_macro_std=scores["test_f1_macro"].std(),
        map_at_3_mean=scores["test_map_at_3"].mean(),
        map_at_3_std=scores["test_map_at_3"].std(),
        fold_scores=scores,
    )


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.model_name,
                "f1_macro_mean": r.f1_macro_mean,
                "f1_macro_std": r.f1_macro_std,
                "map_at_3_mean": r.map_at_3_mean,
                "map_at_3_std": r.map_at_3_std,
            }
            for r in results
        ]
    ).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
