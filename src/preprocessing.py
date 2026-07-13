import numpy as np
import wfdb
from scipy.signal import butter, filtfilt


def filtrar_señal(señal, fs=100, lowcut=0.5, highcut=40.0, orden=4):
    """
    Aplica un filtro paso-banda para limpiar la señal:
    - Elimina el 'baseline wander' (desplazamiento de la línea base),
      causado por la respiración y el movimiento del paciente
      (son frecuencias muy bajas, por debajo de 0.5 Hz).
    - Elimina ruido de alta frecuencia, como interferencias eléctricas
      o temblor muscular (por encima de 40 Hz).
    """
    nyquist = 0.5 * fs
    b, a = butter(orden, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, señal, axis=0)


def normalizar_z_score(señal):
    """
    Normaliza cada derivación (columna) para que tenga media 0 y
    desviación estándar 1. Esto evita que derivaciones con amplitudes
    naturalmente más grandes dominen el aprendizaje de la red neuronal.
    """
    media = señal.mean(axis=0)
    std = señal.std(axis=0)
    std[std == 0] = 1e-8  # evita dividir entre cero si una derivación es plana
    return (señal - media) / std


def procesar_ecg(ruta_archivo, fs=100):
    """
    Pipeline completo para un solo ECG: carga, filtra y normaliza.
    Devuelve None si algo falla, para poder descartar ese registro
    de forma segura sin romper todo el proceso.
    """
    try:
        record = wfdb.rdrecord(ruta_archivo)
        señal = record.p_signal

        if np.isnan(señal).any():
            return None

        señal = filtrar_señal(señal, fs=fs)
        señal = normalizar_z_score(señal)
        return señal.astype(np.float32)
    except Exception:
        return None