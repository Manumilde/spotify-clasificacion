#!/usr/bin/env python3
"""CLI para entrenar y tunear un modelo del registro y guardarlo en disco.

Ejemplos:
    python scripts/train.py --model random_forest
    python scripts/train.py --model logistic_regression --search grid --cv-folds 5
    python scripts/train.py --list-models
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotify_clf import config, data
from spotify_clf.models import get_model_registry
from spotify_clf.tuning import tune_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="random_forest")
    parser.add_argument("--search", choices=["grid", "random"], default="random")
    parser.add_argument("--n-iter", type=int, default=20, help="Solo para --search random")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--train-path", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    registry = get_model_registry()

    if args.list_models:
        print("Modelos disponibles:", ", ".join(registry.keys()))
        return

    if args.model not in registry:
        parser.error(
            f"Modelo '{args.model}' no encontrado. Opciones: {', '.join(registry.keys())}"
        )

    spec = registry[args.model]

    print(f"Cargando datos de entrenamiento desde {args.train_path} ...")
    train_df = data.load_train(args.train_path)
    X, y = data.get_features_and_target(train_df)

    print(
        f"Tuneando '{spec.name}' con {args.search} search "
        f"({args.cv_folds} folds, scoring=f1_macro) ..."
    )
    search = tune_model(
        spec,
        X,
        y,
        search_type=args.search,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
    )

    print(f"Mejor F1-Macro (CV): {search.best_score_:.4f}")
    print(f"Mejores hiperparámetros: {search.best_params_}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MODELS_DIR / f"{spec.name}.joblib"
    joblib.dump(search.best_estimator_, out_path)
    print(f"Modelo guardado en {out_path}")


if __name__ == "__main__":
    main()
