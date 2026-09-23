"""Paths y nombres de columnas del dataset.

Si la competencia de Kaggle usa nombres de columnas distintos a los del
dataset original (por ejemplo `id` en vez de `track_id`), alcanza con
ajustar las constantes de acá.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "outputs" / "submissions"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
REFERENCE_PATH = DATA_RAW_DIR / "dataset_reference.csv"

# Nombres de columnas
ID_COL = "track_id"
TARGET_COL = "track_genre"

# Columnas de texto / identificadoras: no se usan como features numéricas
# directas. Se dropean por default en el preprocesamiento (ver
# preprocessing.py); dejar documentado acá para que sea fácil de cambiar.
TEXT_COLS = ["artists", "album_name", "track_name"]

# Features categóricas de baja cardinalidad (se codifican con one-hot)
CATEGORICAL_COLS = ["key", "mode", "time_signature", "explicit"]

# Features numéricas continuas (se escalan)
NUMERIC_COLS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

RANDOM_STATE = 42
