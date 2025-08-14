# Titanic Survival API (FastAPI + Render)
Servicio de predicción (Regresión Logística) que estima la probabilidad de supervivencia de un pasajero del Titanic. La API acepta **entradas crudas**, tal como vienen en el dataset, y realiza el preprocesamiento internamente (imputación, One‑Hot Encoding y `familysize`).

<!-- ---
## 1) Demo en vivo
- **URL (Render)**: https://TU-APP.onrender.com
- **Docs (Swagger)**: https://TU-APP.onrender.com/docs -->

> Si ves error al primer intento, espera unos segundos y reintenta (posible *cold start* en plan Free).

---
## 2) Estructura del proyecto
```
titanic-api/
├─ api/
│  ├─ main.py                # FastAPI app y endpoints
│  ├─ schemas.py             # modelos Pydantic (entrada/salida)
│  └─ __init__.py            # hace que api sea paquete
├─ model/
│  ├─ train_model.py         # entrena y guarda pipeline.joblib
│  ├─ pipeline.joblib        # modelo + preprocesamiento (versionado)
│  └─ meta.json              # metadatos (features/categorías)
├─ client.py                 # script de pruebas (3 requests)
├─ client.ipynb              # mismo script cliente.py pero con los output
├─ requirements.txt          # dependencias mínimas para runtime
├─ render.yaml               # blueprint para Render
├─ README.md                 # documentación
├─ .gitignore                # exclusiones git
└─ data/                     # (opcional) titanic.csv si no usas seaborn
```

---
## 3) Endpoints
- `GET /` → Índice con rutas útiles.
- `GET /health` → Chequeo rápido de liveness. Responde `{ "status": "ok" }`.
- `GET /ready` → (opcional) Readiness: confirma que el modelo está cargado. Responde `{ "ready": true }` o `503` si no.
- `POST /predict` → **Endpoint principal**. Recibe un JSON con los datos del pasajero y entrega `{prediction, probability, verdict}`.

---
## 4) Esquema de entrada (JSON)
Ejemplo:
```json
{
  "pclass": 3,
  "age": 29,
  "fare": 7.25,
  "sibsp": 0,
  "parch": 0,
  "sex": "male",
  "embarked": "S"
}
```
**Campos:**
- `pclass` *(int, requerido)*: 1, 2 o 3.
- `age` *(float, opcional)*: ≥ 0. Si no viene o es `null`, se imputa (mediana).
- `fare` *(float, opcional)*: ≥ 0. Si no viene o es `null`, se imputa (mediana).
- `sibsp` *(int, requerido)*: ≥ 0. N° de hermanos/esposo(a) a bordo.
- `parch` *(int, requerido)*: ≥ 0. N° de padres/hijos a bordo.
- `sex` *(enum, requerido)*: `"male"` | `"female"` (normaliza a minúsculas).
- `embarked` *(enum, requerido)*: `"C"` | `"Q"` | `"S"` (normaliza a mayúsculas).

**Nota:** La API **deriva** internamente `familysize = sibsp + parch + 1` y aplica el mismo preprocesamiento usado al entrenar.

---
## 5) Esquema de salida (JSON)
```json
{
  "prediction": 1,
  "probability": 0.8731,
  "verdict": "sobrevive"
}
```
- `prediction` *(int)*: `0 = no sobrevive`, `1 = sobrevive`.
- `probability` *(float)*: probabilidad de clase 1 en rango `[0, 1]`.
- `verdict` *(string)*: etiqueta humana derivada de `prediction` (`"sobrevive"` o `"no_sobrevive"`).

```json
{
  "prediction": 1,
  "probability": 0.8731
}
```
- `prediction` *(int)*: `0 = no sobrevive`, `1 = sobrevive`.
- `probability` *(float)*: probabilidad de clase 1 en rango `[0, 1]`.
