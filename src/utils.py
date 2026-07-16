import random
import numpy as np
import torch


def fijar_semilla(semilla=42):
    """
    Fija todas las fuentes de números aleatorios del proyecto (Python, 
    NumPy y PyTorch) para que el entrenamiento sea reproducible: ejecutar 
    el mismo código dos veces debe dar exactamente el mismo resultado.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)