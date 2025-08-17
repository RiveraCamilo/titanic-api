# Titanic Survival API (FastAPI + Render)
Servicio de predicción (Regresión Logística) que estima la probabilidad de supervivencia de un pasajero del Titanic. La API acepta **entradas crudas**, tal como vienen en el dataset, y realiza el preprocesamiento internamente (imputación, One‑Hot Encoding y creación de variable `familysize`).

---
## 1) Estructura del proyecto
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
## 2) Demo en vivo
- **URL (Render)**: https://titanic-api-vtf1.onrender.com/
- **Docs (Swagger)**: https://titanic-api-vtf1.onrender.com/docs

> Si ves error al primer intento, espera unos segundos y reintenta (posible *cold start* en plan Free).

---
## 3) Endpoints
- `GET /` → Índice con rutas disponibles.
- `GET /health` → Chequeo rápido de liveness. Responde `{ "status": "ok" }`.
- `GET /ready` → Readiness: confirma que el modelo está cargado. Responde `{ "ready": true }` o `503` si no.
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
- `pclass` *(int, requerido)*: 1, 2 o 3. **Clase del pasajero**.
- `age` *(float, opcional)*: ≥ 0. **Edad del pasajero** - Si no viene o es `null`, se imputa (mediana).
- `fare` *(float, opcional)*: ≥ 0. **Tarifa pagada** - Si no viene o es `null`, se imputa (mediana).
- `sibsp` *(int, requerido)*: ≥ 0. **N° de hermanos/esposo(a) a bordo**.
- `parch` *(int, requerido)*: ≥ 0. **N° de padres/hijos a bordo**.
- `sex` *(enum, requerido)*: `"male"` | `"female"`. **Sexo del pasajero** (normaliza a minúsculas).
- `embarked` *(enum, requerido)*: `"C"` | `"Q"` | `"S"`. **Puerto de embarque** (normaliza a mayúsculas).

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
- `verdict` *(string)*: etiqueta de texto derivada de `prediction` (`"sobrevive"` o `"no_sobrevive"`).
---
## 6) Ejemplos de uso
### cURL
```bash
curl -X POST "https://titanic-api-vtf1.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "pclass":3, "age":29, "fare":7.25,
        "sibsp":0, "parch":0,
        "sex":"male", "embarked":"S"
      }'
```

### Python (requests)
```python
import requests
API = "https://titanic-api-vtf1.onrender.com/predict"
payload = {
  "pclass": 1, "age": 38, "fare": 71.2833,
  "sibsp": 1, "parch": 0,
  "sex": "female", "embarked": "C"
}
print(requests.post(API, json=payload, timeout=20).json())
```

### Postman
- Método: **POST**
- URL: `https://titanic-api-vtf1.onrender.com/predict`
- Headers: `Content-Type: application/json`
- Body → raw (JSON) → pega el ejemplo de arriba.

---
## 7) Correr localmente
```bash
# 1) Instalar dependencias (mínimas para producción)
pip install -r requirements.txt

# 2) Entrenar (opcional si ya versionaste el .joblib)
python model/train_model.py   # genera model/pipeline.joblib

# 3) Levantar API local
uvicorn api.main:app --reload  # http://127.0.0.1:8000/docs
```
**Cliente de prueba**:
```bash
# contra local
python client.py

# contra Render
API_URL="https://titanic-api-vtf1.onrender.com/" python client.py
```

**Tests (opcional):**
```bash
python -m pip install -r requirements-dev.txt  # si usas archivo dev
python -m pytest -q
```

---
## 8) Errores comunes y respuestas
### Validación (422 Unprocessable Entity)
Ejemplo: `embarked` inválido
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body","embarked"],
      "msg": "embarked debe ser 'C', 'Q' o 'S'",
      "input": "X"
    }
  ]
}
```
### Error de predicción (400 Bad Request)
```json
{ "detail": "Error en la predicción: <mensaje>" }
```
### Readiness (503 Service Unavailable)
```json
{ "detail": "not ready: ..." }
```

---
## 9) Despliegue en Render (resumen)
- **Build**: `pip install -r requirements.txt`
- **Start**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Asegúrate de **versionar** `model/pipeline.joblib` en el repo para no entrenar en Render.
- Opcional: `render.yaml` con `runtime: python` y plan `free`.

---
## 10) Detalles del modelo
- Algoritmo: `LogisticRegression(max_iter=1000)`.
- Preprocesamiento: `SimpleImputer(strategy="median")` para numéricos y `OneHotEncoder` para `sex` y `embarked` (`handle_unknown="ignore"`).
- Features de entrada: `pclass, age, fare, familysize, sex, embarked`.
- Metadata: ver `model/meta.json`.

---
## 11) Troubleshooting
- **`Error: Option '--port' requires an argument`**: localmente usa `--port 8000` (Render inyecta `$PORT`).
- **`No se pudo cargar el modelo...`**: falta `model/pipeline.joblib` → ejecuta `python model/train_model.py` o versiona el archivo en el repo.
- **`from api.main import app` falla en tests**: corre `pytest` desde la raíz y asegúrate de tener `api/__init__.py`.
- **Diferencias de versiones**: sincroniza `requirements.txt` con lo instalado localmente.

---
## 12) Licencia & contacto
- Uso académico/demo.
- Contacto: *ciriverav@gmail.com*.