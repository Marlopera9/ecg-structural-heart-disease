import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from src.gradcam import GradCAM1D
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

capa_objetivo = modelo.bloque3[2]
gradcam = GradCAM1D(modelo, capa_objetivo)


class ExplicacionInput(BaseModel):
    señal: list       # ECG crudo, forma (1000, 12), sin preprocesar
    indice_clase: int # 0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP


@app.post("/explicar")
def explicar(datos: ExplicacionInput):
    señal_cruda = np.array(datos.señal, dtype=np.float32)
    señal_filtrada = filtrar_señal(señal_cruda)
    señal_normalizada = normalizar_z_score(señal_filtrada)

    tensor = torch.tensor(señal_normalizada, dtype=torch.float32).transpose(0, 1)
    cam, probabilidad = gradcam.generar(tensor, datos.indice_clase)

    return {
        "cam": cam.tolist(),
        "probabilidad": round(probabilidad, 4),
        "señal_preprocesada": señal_normalizada.tolist()  # para poder graficarla ya limpia
    }

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

class ECGPreprocesadoInput(BaseModel):
    señal: list  # YA preprocesada (filtrada + normalizada), forma (1000, 12)


@app.post("/predecir_preprocesado")
def predecir_preprocesado(datos: ECGPreprocesadoInput):
    señal = np.array(datos.señal, dtype=np.float32)
    tensor = torch.tensor(señal, dtype=torch.float32).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.sigmoid(logits)[0].tolist()

    return {clase: round(prob, 4) for clase, prob in zip(CLASES, probabilidades)}


@app.post("/explicar_preprocesado")
def explicar_preprocesado(datos: ExplicacionInput):
    señal = np.array(datos.señal, dtype=np.float32)
    tensor = torch.tensor(señal, dtype=torch.float32).transpose(0, 1)
    cam, probabilidad = gradcam.generar(tensor, datos.indice_clase)
    return {"cam": cam.tolist(), "probabilidad": round(probabilidad, 4)}