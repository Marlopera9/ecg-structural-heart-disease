import sys
sys.path.append('.')

import numpy as np
from src.preprocessing import filtrar_señal, normalizar_z_score


def test_filtrar_señal_no_produce_nan():
    señal_aleatoria = np.random.randn(1000, 12).astype(np.float32)
    resultado = filtrar_señal(señal_aleatoria)
    assert not np.isnan(resultado).any(), "El filtro no debería generar valores NaN"


def test_normalizar_da_media_cero_y_desviacion_uno():
    señal_aleatoria = np.random.randn(1000, 12).astype(np.float32) * 5 + 3
    resultado = normalizar_z_score(señal_aleatoria)

    media_por_derivacion = resultado.mean(axis=0)
    std_por_derivacion = resultado.std(axis=0)

    assert np.allclose(media_por_derivacion, 0, atol=1e-5), "La media tras normalizar debería ser ~0"
    assert np.allclose(std_por_derivacion, 1, atol=1e-5), "La desviación estándar tras normalizar debería ser ~1"


def test_normalizar_no_falla_con_señal_plana():
    """Una derivación completamente plana (std=0) no debería causar división por cero."""
    señal_plana = np.zeros((1000, 12), dtype=np.float32)
    resultado = normalizar_z_score(señal_plana)
    assert not np.isnan(resultado).any(), "Una señal plana no debería producir NaN al normalizar"