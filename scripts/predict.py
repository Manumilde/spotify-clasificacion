#!/usr/bin/env python3
"""CLI para generar la submission de Kaggle a partir de un modelo guardado.

Ejemplo:
    python scripts/predict.py --model random_forest
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotify_clf import config, data
from spotify_clf.submission import build_submission, save_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="random_forest")
    parser.add_argument("--test-path", type=Path, default=config.TEST_PATH)
    parser.add_argument(
        "--out-path", type=Path, default=config.SUBMISSIONS_DIR / "submission.csv"
    )
    args = parser.parse_args()

    model_path = config.MODELS_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró {model_path}. Corré primero scripts/train.py --model {args.model}"
        )

    print(f"Cargando modelo desde {model_path} ...")
    pipeline = joblib.load(model_path)

    print(f"Cargando datos de test desde {args.test_path} ...")
    test_df = data.load_test(args.test_path)
    X_test = test_df.drop(
        columns=[c for c in [config.ID_COL, *config.TEXT_COLS] if c in test_df.columns]
    )

    predictions = pipeline.predict(X_test)
    submission_df = build_submission(test_df[config.ID_COL], predictions)

    out_path = save_submission(submission_df, args.out_path)
    print(f"Submission guardada en {out_path} ({len(submission_df)} filas)")


if __name__ == "__main__":
    main()
