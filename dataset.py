import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class PairDataset(Dataset):
    """
    Custom Dataset for loading pairs and labels from individual .npy files.
    Each pair consists of two scenarios (scenario_1 and scenario_2) and a corresponding label.
    """
    def __init__(self, pair_dir, label_dir, scaler, file_list=None):
        self.pair_dir = pair_dir
        self.label_dir = label_dir
        self.scaler = scaler if scaler is not None else StandardScaler()

        if file_list is not None:
            self.pair_files = sorted(file_list)
            self.label_files = [f.replace('pair', 'label') for f in self.pair_files]
        else:
            self.pair_files = sorted([
                f for f in os.listdir(pair_dir)
                if f.startswith('pair') and f.endswith('.npy')
            ])
            self.label_files = sorted([
                f for f in os.listdir(label_dir)
                if f.startswith('label') and f.endswith('.npy')
            ])

        assert len(self.pair_files) == len(self.label_files), "Mismatch between number of pairs and labels"

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        pair_path = os.path.join(self.pair_dir, self.pair_files[idx])
        pair = np.load(pair_path, allow_pickle=True)

        scenario_1 = np.array(pair[0], dtype=np.float32)
        scenario_2 = np.array(pair[1], dtype=np.float32)

        # Get sequence lengths before padding
        length_1 = scenario_1.shape[0]
        length_2 = scenario_2.shape[0]

        # Transform the data using the provided scaler
        scenario_1 = self.scaler.transform(scenario_1)
        scenario_2 = self.scaler.transform(scenario_2)

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = np.load(label_path).item()

        scenario_1 = torch.tensor(scenario_1, dtype=torch.float32)
        scenario_2 = torch.tensor(scenario_2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        length_1 = torch.tensor(length_1, dtype=torch.long)
        length_2 = torch.tensor(length_2, dtype=torch.long)

        return scenario_1, scenario_2, label, length_1, length_2
