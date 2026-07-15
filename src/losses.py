import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMultiLabel(nn.Module):
    """
    Focal Loss adaptada a clasificación multi-etiqueta.
    Reduce el peso de aprendizaje de los ejemplos 'fáciles' (que el modelo
    ya clasifica bien) para que el modelo dedique más atención relativa
    a los ejemplos difíciles o pertenecientes a clases minoritarias (ej. HYP).

    Parámetros:
        alpha: controla el balance general entre positivos y negativos.
        gamma: controla cuánto se reduce el peso de los ejemplos fáciles.
               A mayor gamma, más se ignoran los casos ya bien aprendidos.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, etiquetas):
        bce = F.binary_cross_entropy_with_logits(logits, etiquetas, reduction='none')
        probas = torch.sigmoid(logits)

        # p_t: qué probabilidad le dio el modelo a la respuesta CORRECTA
        p_t = probas * etiquetas + (1 - probas) * (1 - etiquetas)
        peso_focal = (1 - p_t) ** self.gamma

        peso_alpha = self.alpha * etiquetas + (1 - self.alpha) * (1 - etiquetas)

        perdida = peso_alpha * peso_focal * bce
        return perdida.mean()