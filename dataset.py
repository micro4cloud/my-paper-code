import torch
from torch.utils.data import Dataset
import numpy as np

class VLCDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = data['features']
        self.y = data['labels']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)