import torch
from tqdm import tqdm


def entrenar_una_epoca(modelo, loader, funcion_perdida, optimizador, dispositivo):
    """Recorre todo el conjunto de entrenamiento una vez, ajustando los
    pesos del modelo. Devuelve la pérdida promedio de la época."""
    modelo.train()
    perdida_total = 0.0

    for señales, etiquetas in tqdm(loader, desc="Entrenando", leave=False):
        señales = señales.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()
        salidas = modelo(señales)
        perdida = funcion_perdida(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * señales.size(0)

    return perdida_total / len(loader.dataset)


def evaluar(modelo, loader, funcion_perdida, dispositivo):
    """Recorre un conjunto de datos SIN ajustar pesos (solo mide qué tan
    bien predice). Devuelve la pérdida promedio, más todas las predicciones
    y etiquetas reales, listas para calcular métricas como AUROC."""
    modelo.eval()
    perdida_total = 0.0
    todas_predicciones = []
    todas_etiquetas = []

    with torch.no_grad():
        for señales, etiquetas in tqdm(loader, desc="Evaluando", leave=False):
            señales = señales.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            salidas = modelo(señales)
            perdida = funcion_perdida(salidas, etiquetas)
            perdida_total += perdida.item() * señales.size(0)

            probas = torch.sigmoid(salidas)
            todas_predicciones.append(probas.cpu())
            todas_etiquetas.append(etiquetas.cpu())

    perdida_promedio = perdida_total / len(loader.dataset)
    predicciones = torch.cat(todas_predicciones).numpy()
    etiquetas_reales = torch.cat(todas_etiquetas).numpy()

    return perdida_promedio, predicciones, etiquetas_reales