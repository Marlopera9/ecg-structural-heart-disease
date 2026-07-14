import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, ruta_X, ruta_y):
        self.X = np.load(ruta_X)
        self.y = np.load(ruta_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        señal = self.X[idx]        # forma: (1000, 12)
        etiqueta = self.y[idx]     # forma: (5,)

        # PyTorch espera (canales, longitud), no (longitud, canales)
        señal_tensor = torch.tensor(señal, dtype=torch.float32).transpose(0, 1)
        etiqueta_tensor = torch.tensor(etiqueta, dtype=torch.float32)

        return señal_tensor, etiqueta_tensor