# Spotify - Clasificación de Género Musical (TD6 - TP2)

Estructura de proyecto para el TP2 de Tecnología Digital VI (clasificación
multiclase de género musical a partir de métricas acústicas de Spotify,
competencia interna de Kaggle).

## 1. Setup local (VS Code)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Seleccioná el intérprete `.venv` en VS Code (`Ctrl+Shift+P` → *Python: Select
Interpreter*) para que el notebook y los scripts corran con las mismas
dependencias.

## 2. Datos

Los archivos de la competencia **no se versionan** en el repo (ver
`.gitignore`) porque son propios de Kaggle. Descargalos de la competencia y
colocalos así:

```
data/raw/train.csv
data/raw/test.csv
```

`data/raw/` incluye un `dataset.csv` de referencia (opcional, mismo esquema
de columnas que el dataset original de Spotify) solo para explorar el
esquema si todavía no tenés el `train.csv` de la competencia — no es el
dataset de la competencia y no debe usarse para entrenar el modelo final.

Columnas esperadas (ver `src/spotify_clf/config.py` para ajustar nombres si
la competencia usa otros):

- `track_id`, `artists`, `album_name`, `track_name` (identificadores/texto)
- `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`,
  `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`,
  `liveness`, `valence`, `tempo`, `time_signature` (features)
- `track_genre` (target, solo en train)

Si la competencia usa nombres de columnas distintos (p. ej. `id` en vez de
`track_id`, o el target con otro nombre), actualizá `ID_COL` / `TARGET_COL`
en `src/spotify_clf/config.py` — el resto del código no necesita tocarse.

## 3. Estructura

```
data/
  raw/            # train.csv, test.csv (no versionados)
  processed/      # artefactos intermedios (no versionados)
notebooks/
  TP2_spotify_clasificacion.ipynb   # notebook entregable (EDA, modelos, informe)
src/spotify_clf/
  config.py         # paths y nombres de columnas
  data.py           # carga de datos y split train/val
  preprocessing.py  # ColumnTransformer (escalado + one-hot)
  metrics.py        # F1-macro y MAP@3 (+ scorers de sklearn)
  models.py         # registro de modelos y grillas de hiperparámetros
  tuning.py         # wrapper de GridSearchCV/RandomizedSearchCV
  evaluate.py       # cross-validation, comparación F1-macro vs MAP@3
  submission.py     # generación del CSV para subir a Kaggle
scripts/
  train.py          # entrena + tunea + guarda el mejor modelo (CLI)
  predict.py        # carga el modelo guardado y genera la submission (CLI)
outputs/
  models/           # modelos entrenados (.joblib, no versionados)
  submissions/      # CSVs de submission (no versionados)
```

## 4. Flujo de trabajo

### Opción A: notebook (recomendado para EDA + informe)

Abrí `notebooks/TP2_spotify_clasificacion.ipynb` en VS Code y corré las
celdas en orden. El notebook importa las funciones de `src/spotify_clf`,
así que la lógica pesada vive en el paquete y el notebook queda enfocado en
análisis, resultados y discusión (para el informe).

### Opción B: scripts (para iterar rápido / reproducir la submission)

```bash
# entrena todos los modelos registrados, hace tuning de hiperparámetros
# optimizando F1-macro, elige el mejor por CV y lo guarda en outputs/models/
python scripts/train.py --model random_forest

# genera outputs/submissions/submission.csv a partir del modelo guardado
python scripts/predict.py --model random_forest
```

Ver `python scripts/train.py --help` para las opciones (modelo, búsqueda
Grid/Randomized, cantidad de folds, etc.).

## 5. Métricas

- **F1-Macro**: métrica oficial de la competencia de Kaggle (predicción
  top-1). Se usa como `scoring` en la búsqueda de hiperparámetros y en la
  selección de modelo.
- **MAP@3**: implementada en `src/spotify_clf/metrics.py`
  (`mean_average_precision_at_k`), se calcula sobre el conjunto de
  validación local a partir de las probabilidades (`predict_proba`) de cada
  modelo, para comparar con F1-Macro y discutir el trade-off de negocio
  (ver notebook, sección final).

## 6. Reproducir el CSV final enviado a Kaggle

1. Colocar `train.csv`/`test.csv` en `data/raw/`.
2. Correr el notebook de punta a punta (o `scripts/train.py` +
   `scripts/predict.py` con el modelo elegido).
3. El CSV queda en `outputs/submissions/submission.csv`, listo para subir a
   Kaggle.
