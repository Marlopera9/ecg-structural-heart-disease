import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import ECG_CNN1D
from src.preprocessing import filtrar_señal, normalizar_z_score

app = FastAPI(
    title="API de Detección de Enfermedad Cardíaca Estructural",
    description="Predice la probabilidad de 5 categorías diagnósticas a partir de un ECG de 12 derivaciones.",
)

CLASES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

modelo = ECG_CNN1D(num_leads=12, num_classes=5)
modelo.load_state_dict(torch.load('models/mejor_modelo.pt', map_location='cpu', weights_only=True))
modelo.eval()


class ECGInput(BaseModel):
    señal: list  # 1000 listas de 12 valores cada una: [tiempo][derivación], SIN preprocesar


@app.get("/")
def raiz():
    return {"mensaje": "API activa. Envía un ECG mediante POST a /predecir"}


@app.post("/predecir")
def predecir(datos: ECGInput):
    señal_cruda = np.array(datos.señal, dtype=np.float32)

    señal_filtrada = filtrar_señal(señal_cruda)
    señal_normalizada = normalizar_z_score(señal_filtrada)

    tensor = torch.tensor(señal_normalizada, dtype=torch.float32).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.sigmoid(logits)[0].tolist()

    return {clase: round(prob, 4) for clase, prob in zip(CLASES, probabilidades)}