# distance_dataset.py

import numpy as np
from torch.utils.data import Dataset

class DistanceDataset(Dataset):
    def __init__(self, distances_file, labels_file):
        self.distances = np.load(distances_file)
        self.labels = np.load(labels_file)
        
    def __len__(self):
        return len(self.distances)
    
    def __getitem__(self, idx):
        distance = self.distances[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Reshape to (1,) to match input expectations
        distance = np.array([distance])
        label = np.array([label])
        
        return distance, label
