import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from api.schemas import PredictRequest, PredictResponse

APP_NAME = "Titanic Survival API"
app = FastAPI(title=APP_NAME, version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "model/pipeline.joblib")

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}")

@app.get("/", tags=["root"]) # Home con los endpoints
def root(request: Request):
    return {
        "service": APP_NAME,
        "version": app.version,
        "health": str(request.url_for("health")),
        "readiness": str(request.url_for("ready")),
        "docs": "/docs",
        "predict": str(request.url_for("predict")),
    }

@app.get("/health", tags=["health"])  # Para confirmacion rápida de la api
def health():
    return {"status": "ok"}

@app.get("/ready") # Para confirmacion de que puede predecir
def ready():
    try:
        # Señal mínima de que el modelo cargó y está entrenado
        _ = pipeline.classes_
        return {"ready": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No está listo: {e}")

@app.post("/predict", response_model=PredictResponse, tags=["predict"])  
def predict(req: PredictRequest):
    try:
        # Calcula feature derivada
        familysize = req.sibsp + req.parch + 1

        # DataFrame con las columnas que espera el pipeline
        X = pd.DataFrame([
            {
                "pclass": req.pclass,
                "age": req.age,
                "fare": req.fare,
                "familysize": familysize,
                "sex": req.sex,
                "embarked": req.embarked,
            }
        ])

        # Predicción
        proba = float(pipeline.predict_proba(X)[0][1])
        pred = int(pipeline.predict(X)[0])
        return {"prediction": pred, "probability": proba}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {e}")