import sys
sys.path.append('.')

from fastapi.testclient import TestClient
from src.api import app

cliente = TestClient(app)


def test_endpoint_raiz_responde():
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200


def test_endpoint_predecir_devuelve_5_clases():
    señal_de_prueba = [[0.0] * 12 for _ in range(1000)]  # ECG "silencio", solo para probar la forma
    respuesta = cliente.post("/predecir", json={"señal": señal_de_prueba})

    assert respuesta.status_code == 200
    resultado = respuesta.json()
    assert len(resultado) == 5
    assert all(0.0 <= prob <= 1.0 for prob in resultado.values())