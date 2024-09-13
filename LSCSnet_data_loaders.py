import os
import glob
import csv
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = glob.glob(os.path.join(directory, '*CSV'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([float(x) for x in row])
        data = torch.tensor(data, dtype=torch.float32)
        data = data / data.max(dim=-1, keepdim=True).values
        data = data[:, 1].unsqueeze(0)  # Add channel dimension
        if self.transform:
            data = self.transform(data)
        return data
