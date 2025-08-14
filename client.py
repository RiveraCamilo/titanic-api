import os
import json
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"

cases = [
    {
        "pclass": 3,
        "age": 29,
        "fare": 7.25,
        "sibsp": 0,
        "parch": 0,
        "sex": "male",
        "embarked": "S"
    },
    {
        "pclass": 1,
        "age": 38,
        "fare": 71.2833,
        "sibsp": 1,
        "parch": 0,
        "sex": "female",
        "embarked": "C"
    },
    {
        "pclass": 2,
        "age": 21,
        "fare": 13.0,
        "sibsp": 0,
        "parch": 1,
        "sex": "female",
        "embarked": "Q"
    }
]

for i, payload in enumerate(cases, 1):
    r = requests.post(PREDICT_URL, json=payload, timeout=20)
    print(f"\n--- Caso {i} ---")
    print("POST", PREDICT_URL)
    print("Enviado:", json.dumps(payload, ensure_ascii=False))
    print("Status:", r.status_code)
    try:
        print("Respuesta:", r.json())
    except Exception:
        print("Respuesta no-JSON:", r.text)