# seqential.py

import torch
import torch.nn as nn

class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.ReLU(),

            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)
