"""
Entrena un pipeline de Regresión Logística para Titanic y guarda model/pipeline.joblib.
- Fuente de datos:
1) Si existe data/titanic.csv, la usa.
2) Si no, intenta cargar seaborn.load_dataset('titanic').

Requisitos:
- scikit-learn, pandas, seaborn.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = "data/titanic.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Carga de datos
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    try:
        import seaborn as sns  # solo para cargar dataset ejemplo
        df = sns.load_dataset("titanic")
    except Exception as e:
        raise RuntimeError(
            "No se encontró data/titanic.csv y falló cargar seaborn.titanic. "
            "Descarga un CSV a data/titanic.csv o instala seaborn para usar el dataset de ejemplo."
        )

# Selección y limpieza mínima
cols = ["survived", "pclass", "age", "fare", "sibsp", "parch", "sex", "embarked"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Faltan columnas en el dataset: {missing}")

df = df[cols].copy()

# Normalizaciones básicas
# Convertimos a 'object' para poder aplicar .str solo a no nulos y mantener NaN como NaN
import numpy as np

df["sex"] = df["sex"].astype(object)
df["embarked"] = df["embarked"].astype(object)
# Normaliza solo donde hay valor, evitando convertir NaN en string "nan"
df.loc[df["sex"].notna(), "sex"] = (
    df.loc[df["sex"].notna(), "sex"].astype(str).str.strip().str.lower()
)
df.loc[df["embarked"].notna(), "embarked"] = (
    df.loc[df["embarked"].notna(), "embarked"].astype(str).str.strip().str.upper()
)

# Feature derivada
df["familysize"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1

# Conjunto de features finales que verá el modelo
feature_cols = ["pclass", "age", "fare", "familysize", "sex", "embarked"]
X = df[feature_cols].copy()
y = df["survived"].astype(int)

# Preprocesamiento
num_features = ["pclass", "age", "fare", "familysize"]
cat_features = ["sex", "embarked"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat",
            OneHotEncoder(
                categories=[["female", "male"], ["C", "Q", "S"]],
                handle_unknown="ignore",
                sparse_output=False,
            ),
        cat_features),
    ],
    remainder="drop",
)

clf = LogisticRegression(max_iter=1000)
pipe = Pipeline(steps=[("pre", preprocess), ("clf", clf)])

# Entrena
pipe.fit(X, y)

# Guarda modelo
joblib.dump(pipe, MODEL_PATH)

# Guarda metadata útil
meta = {
    "features_raw": feature_cols,
    "categories": {
        "sex": ["female", "male"],
        "embarked": ["C", "Q", "S"],
    }
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Modelo guardado en {MODEL_PATH}")