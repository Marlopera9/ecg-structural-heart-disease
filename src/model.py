import torch.nn as nn


class ECG_CNN1D(nn.Module):
    """
    Red neuronal convolucional 1D para clasificar ECGs de 12 derivaciones
    en 5 categorías diagnósticas (multi-etiqueta): NORM, MI, STTC, CD, HYP.
    """

    def __init__(self, num_leads=12, num_classes=5):
        super().__init__()

        self.bloque1 = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bloque2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bloque3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.clasificador(x)
        return x