import torch
import numpy as np


class GradCAM1D:
    """
    Implementación de Grad-CAM adaptada a señales 1D (en vez de imágenes 2D).
    Genera un mapa de calor mostrando qué instantes de tiempo de la señal
    influyeron más en la predicción del modelo para una clase concreta.
    """

    def __init__(self, modelo, capa_objetivo):
        self.modelo = modelo
        self.capa_objetivo = capa_objetivo
        self.activaciones = None
        self.gradientes = None

        capa_objetivo.register_forward_hook(self._guardar_activaciones)
        capa_objetivo.register_full_backward_hook(self._guardar_gradientes)

    def _guardar_activaciones(self, module, entrada, salida):
        self.activaciones = salida.detach()

    def _guardar_gradientes(self, module, grad_entrada, grad_salida):
        self.gradientes = grad_salida[0].detach()

    def generar(self, señal_tensor, indice_clase):
        """
        señal_tensor: tensor de forma (12, 1000), ya preprocesado.
        indice_clase: qué clase (0=NORM, 1=MI, ... 4=HYP) queremos explicar.
        Devuelve: (mapa_de_calor de longitud 1000, probabilidad predicha)
        """
        self.modelo.zero_grad()

        entrada = señal_tensor.unsqueeze(0)  # añade dimensión de batch: (1, 12, 1000)
        logits = self.modelo(entrada)
        score = logits[0, indice_clase]
        score.backward()

        # Peso de cada canal: promedio del gradiente a lo largo del tiempo
        pesos = self.gradientes.mean(dim=2, keepdim=True)

        # Combinación ponderada de los mapas de activación
        cam = (pesos * self.activaciones).sum(dim=1).squeeze()
        cam = torch.relu(cam)  # solo nos interesa lo que "empuja a favor" de la clase
        cam = cam / (cam.max() + 1e-8)  # normalizamos a [0, 1]
        cam = cam.cpu().numpy()

        # La capa objetivo tiene menos resolución temporal que la señal original
        # (por los MaxPooling), así que interpolamos para que encajen
        cam_interpolado = np.interp(
            np.linspace(0, len(cam) - 1, señal_tensor.shape[1]),
            np.arange(len(cam)),
            cam
        )

        probabilidad = torch.sigmoid(logits[0, indice_clase]).item()
        return cam_interpolado, probabilidad