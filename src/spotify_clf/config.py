"""Paths y nombres de columnas del dataset.

Ajustado a la competencia real del TP2 (dataset anonimizado por la
cátedra, 6 géneros): `train.csv` trae las features + la columna de
género; `test.csv` trae las features + una columna `Id` sintética
(secuencial, generada solo para que Kaggle evalúe). No hay `track_id`,
`artists`, `album_name` ni `track_name` (fueron eliminados al anonimizar).

Si al abrir su `train.csv`/`test.csv` real los nombres de columnas no
coinciden con lo de acá (por ejemplo si el target no se llama
`track_genre`), alcanza con ajustar las constantes de este archivo — el
resto del código no necesita tocarse.
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

# Nombre de la columna id en test.csv (solo test.csv la trae; en train.csv
# no debería existir). Verificar mayúscula/minúscula ("Id") contra el
# archivo real una vez descargado.
ID_COL = "Id"

# Nombre de la columna de género en train.csv. TODO: confirmar contra el
# train.csv real de la competencia (la cátedra no lo detalla en la
# consigna) y ajustar acá si es distinto (ej. "genre", "Genre").
TARGET_COL = "track_genre"

# Nombre de columna que exige el formato de submission de Kaggle para esta
# competencia (ver consigna: "Id,Expected"). Distinto de TARGET_COL a
# propósito: adentro del proyecto trabajamos con TARGET_COL, y solo al
# armar el CSV final se renombra a lo que pide Kaggle.
SUBMISSION_TARGET_COL = "Expected"

# El dataset fue anonimizado por la cátedra: no hay columnas de texto
# libre (track_id/artists/album_name/track_name ya no existen). Se deja
# la lista vacía; get_features_and_target() no rompe si alguna de estas
# columnas no está presente.
TEXT_COLS: list[str] = []

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
