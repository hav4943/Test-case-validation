import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import itertools
import pickle

# Load your pickle file (Replace 'your_data.pkl' with the actual file path)
with open('C:\\Users\\UZEX5M4\\Pickle_data\\All_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure your data is loaded as a DataFrame
if not isinstance(data, pd.DataFrame):
    raise ValueError("Loaded data is not a Pandas DataFrame!")

# Check if 'Label' column exists in the DataFrame
if 'Label' not in data.columns:
    raise ValueError("The DataFrame does not contain a 'Label' column!")


class FullDataset(Dataset):
    def __init__(self, data):
        self.features = data.iloc[:, :-1].values  # All columns except last one are features
        self.labels = data['Label'].values        # The 'Label' column

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Dataset object to manage the data
dataset = FullDataset(data)

# Create pairs lazily for the whole dataset
def create_full_pairs(dataset):
    positive_pairs = []
    negative_pairs = []
    
    # Generate all combinations (i.e., all possible pairs) from the dataset
    for i, (features_i, label_i) in enumerate(dataset):
        for j, (features_j, label_j) in enumerate(dataset):
            if i >= j:
                continue  # To avoid duplicate pairs and self-pairing
            
            pair = (features_i.numpy(), features_j.numpy())
            if label_i == label_j:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)
    
    return positive_pairs, negative_pairs

# Generating pairs across the entire dataset
positive_pairs, negative_pairs = create_full_pairs(dataset)

# Show the result of positive and negative pairs
print(f"Total Positive Pairs: {len(positive_pairs)}")
print(f"Total Negative Pairs: {len(negative_pairs)}")

# Display first 5 pairs for brevity
print("\nFirst 5 Positive Pairs:")
for p in positive_pairs[:5]:
    print(p)

print("\nFirst 5 Negative Pairs:")
for n in negative_pairs[:5]:
    print(n)
