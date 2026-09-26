"""Paths y nombres de columnas del dataset.

Según la descripción oficial del dataset de la competencia: `train.csv`
trae las features acústicas + la columna objetivo `Expected` (el género,
6 valores posibles); `test.csv` trae las mismas features + una columna
`Id` (numérica, secuencial, generada por la cátedra solo para el cruce de
predicciones en Kaggle, sin valor predictivo) y no trae `Expected`. No hay
`track_id`, `artists`, `album_name` ni `track_name` (dataset anonimizado).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "outputs" / "submissions"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_RAW_DIR / "sample_submission.csv"
REFERENCE_PATH = DATA_RAW_DIR / "dataset_reference.csv"

# Id numérico secuencial, solo en test.csv, sin valor predictivo (no usar
# como feature).
ID_COL = "Id"

# Columna objetivo, tal cual la define la cátedra: presente en train.csv,
# ausente en test.csv, y es también el nombre de columna que exige el
# formato de submission de Kaggle ("Id,Expected").
TARGET_COL = "Expected"
SUBMISSION_TARGET_COL = TARGET_COL

# El dataset fue anonimizado por la cátedra: no hay columnas de texto
# libre (track_id/artists/album_name/track_name ya no existen). Se deja
# la lista vacía; get_features_and_target() no rompe si alguna de estas
# columnas no está presente.
TEXT_COLS: list[str] = []

# Features categóricas de baja cardinalidad (se codifican con one-hot).
# Nota: `key` puede valer -1 cuando Spotify no detectó tonalidad; con
# OneHotEncoder eso simplemente se trata como un nivel más, no requiere
# manejo especial.
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
