import torch
from torch.utils.data import Dataset
import os
import numpy as np

class PronunciationDataset(Dataset):
    def __init__(self, feature_dir, label_file):
        self.data = []
        self.feature_dir = feature_dir
        with open(label_file, 'r') as f:
            for line in f:
                audio_id, label = line.strip().split(',')
                self.data.append((audio_id, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id, label = self.data[idx]
        features = np.load(os.path.join(self.feature_dir, f"{audio_id}.npy"))
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
