# Spotify - Clasificación de Género Musical (TD6 - TP2)

Estructura de proyecto para el TP2 de Tecnología Digital VI: clasificación
multiclase de género musical (**6 géneros**) a partir de métricas
acústicas de Spotify, competencia interna de Kaggle armada por la
cátedra.

**Sobre la competencia**: se accede con el link del Campus Virtual (no es
un dataset público de Kaggle) — hay que unirse ("Join Competition") y
descargar `train.csv`/`test.csv` desde la pestaña "Data" de esa
competencia. El dataset fue **anonimizado**: no tiene `track_id`,
`artists`, `album_name` ni `track_name`, y las filas están en orden
aleatorio. `test.csv` trae una columna `Id` sintética (secuencial, solo
para que Kaggle empareje las predicciones); esa columna no está en
`train.csv`. Formato de submission exigido: dos columnas, `Id,Expected`
(ver sección 6). Límite: **5 envíos por día por equipo**.

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

`data/raw/dataset_reference.csv` es el dataset original completo (114
géneros, sin anonimizar) — sirve solo para explorar el esquema de las
features acústicas si todavía no tenés el `train.csv` de la competencia.
**No es el dataset del TP** y no debe usarse para entrenar el modelo
final (tiene otro número de géneros y otras columnas).

Columnas esperadas en los archivos reales de la competencia (ver
`src/spotify_clf/config.py` para ajustar nombres si tu `train.csv`/
`test.csv` difiere):

- `Id`: sintética, secuencial, **solo en `test.csv`** (no está en train).
- `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`,
  `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`,
  `liveness`, `valence`, `tempo`, `time_signature` (features).
- Columna de género (target, solo en `train.csv`) — la consigna no
  especifica su nombre exacto; por default el código asume
  `track_genre` (`config.TARGET_COL`). **Abrí tu `train.csv` y confirmá
  el nombre real de esa columna** antes de correr el notebook; si es
  distinto, cambiá `TARGET_COL` en `src/spotify_clf/config.py` (una sola
  línea, el resto del código no necesita tocarse).

El formato de submission que exige Kaggle (`Id,Expected`) es distinto del
nombre interno del target — está en `config.SUBMISSION_TARGET_COL` y ya
lo arma así `src/spotify_clf/submission.py`.

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

## 6. Cómo subir una submission a Kaggle

1. En la página de la competencia (link del Campus Virtual), pestaña
   **"Submit Predictions"**, subís el CSV generado en
   `outputs/submissions/submission.csv`. Tiene que tener exactamente dos
   columnas: `Id,Expected` (ya lo arma así `submission.py`).
2. Kaggle te devuelve el score (F1-Macro) y tu posición en el
   leaderboard. **Máximo 5 envíos por día por equipo** — usen la
   validación local del notebook (secciones 5-8) para decidir qué modelo
   mandar, no para iterar a ciegas contra Kaggle.
3. Si hay un `sample_submission.csv` en la pestaña "Data" de la
   competencia, úsenlo como referencia exacta del formato esperado.

## 7. Reproducir el CSV final enviado a Kaggle

La consigna pide que el código entregado pueda reproducir exactamente el
CSV que obtuvo el puntaje final:

1. Colocar `train.csv`/`test.csv` en `data/raw/`.
2. Correr el notebook de punta a punta (o `scripts/train.py` +
   `scripts/predict.py` con el modelo elegido) — las semillas están
   fijadas (`config.RANDOM_STATE`) para que el resultado sea reproducible.
3. El CSV queda en `outputs/submissions/submission.csv`, idéntico al que
   se subió a Kaggle para el puntaje final (no lo vuelvan a generar con
   otro modelo/hiperparámetros después de la entrega final).
