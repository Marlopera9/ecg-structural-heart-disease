import sys
sys.path.append('.')

import torch
from src.model import ECG_CNN1D


def test_forma_de_salida_del_modelo():
    modelo = ECG_CNN1D(num_leads=12, num_classes=5)
    entrada_de_prueba = torch.randn(4, 12, 1000)  # batch de 4 ECGs falsos

    salida = modelo(entrada_de_prueba)

    assert salida.shape == (4, 5), f"Se esperaba forma (4, 5), se obtuvo {salida.shape}"


def test_modelo_acepta_longitudes_de_señal_distintas():
    """Verifica que AdaptiveAvgPool1d permite longitudes distintas a 1000
    (relevante si en el futuro se usa con otro dataset, como EchoNext a 2500)."""
    modelo = ECG_CNN1D(num_leads=12, num_classes=5)
    entrada_2500 = torch.randn(2, 12, 2500)

    salida = modelo(entrada_2500)

    assert salida.shape == (2, 5)