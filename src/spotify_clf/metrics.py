"""Métricas de evaluación: F1-Macro (métrica oficial de Kaggle) y MAP@3.

MAP@3 (Mean Average Precision at K=3): para cada canción, se le da crédito
al modelo si el género correcto aparece entre sus top-3 predicciones
(ordenadas por probabilidad), con más peso cuanto más arriba en el ranking
esté. A diferencia de F1-Macro, no exige acertar en la primera predicción.
"""
import numpy as np
from sklearn.metrics import f1_score


def f1_macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def average_precision_at_k(true_label, predicted_labels, k: int = 3) -> float:
    """Average precision @ k para una sola observación.

    `predicted_labels` es una lista de clases ordenada de mayor a menor
    probabilidad (ya truncada o no a k). Devuelve 1/(posición) si
    `true_label` está entre los primeros k, 0 si no.
    """
    predicted_labels = list(predicted_labels)[:k]
    for i, label in enumerate(predicted_labels):
        if label == true_label:
            return 1.0 / (i + 1)
    return 0.0


def mean_average_precision_at_k(y_true, y_pred_ranked, k: int = 3) -> float:
    """MAP@k sobre un conjunto de observaciones.

    Parameters
    ----------
    y_true : array-like de shape (n_samples,)
        Etiqueta verdadera de cada observación.
    y_pred_ranked : array-like de shape (n_samples, n_classes) o lista de
        listas, con las clases predichas ordenadas de mayor a menor
        probabilidad para cada observación (ya truncado a >= k columnas).
    k : int
        Tope de posiciones a considerar (K=3 pedido por la consigna).
    """
    scores = [
        average_precision_at_k(true, ranked, k=k)
        for true, ranked in zip(y_true, y_pred_ranked)
    ]
    return float(np.mean(scores))


def top_k_predictions(model, X, k: int = 3):
    """Dado un modelo con `predict_proba` y `classes_`, devuelve para cada
    fila de X la lista de las top-k clases ordenadas por probabilidad
    descendente.
    """
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    # argsort ascendente -> invertimos para tener las de mayor proba primero
    top_k_idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return classes[top_k_idx]


def map_at_3_from_model(model, X, y_true, k: int = 3) -> float:
    """Calcula MAP@3 evaluando un modelo ya entrenado sobre (X, y_true)."""
    ranked = top_k_predictions(model, X, k=k)
    return mean_average_precision_at_k(y_true, ranked, k=k)


def f1_macro_scorer(estimator, X, y):
    """Scorer compatible con la API de sklearn (`scoring=f1_macro_scorer`)
    para usar F1-Macro en cross_validate / GridSearchCV / RandomizedSearchCV.

    Nota: se implementa como función plana (en vez de
    `sklearn.metrics.make_scorer(f1_score, average="macro")`) porque en
    algunas instalaciones de scikit-learn ese scorer falla silenciosamente
    dentro de la búsqueda paralela (sklearn lo tapa devolviendo NaN en vez
    de propagar el error) — lo cual puede hacer que se "elija" el peor
    modelo sin ningún aviso. Con una función así, si algo sale mal, el
    error se propaga normalmente en vez de convertirse en un NaN silencioso.
    """
    return f1_score(y, estimator.predict(X), average="macro")


def map_at_3_scorer(estimator, X, y):
    """Scorer compatible con la API de sklearn (`scoring=map_at_3_scorer`)
    para usar MAP@3 en cross_validate / GridSearchCV. Requiere que el
    estimator soporte `predict_proba`.
    """
    return map_at_3_from_model(estimator, X, y, k=3)
