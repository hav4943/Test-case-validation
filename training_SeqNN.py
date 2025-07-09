# training_seqNN.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from embedding_dataset import DistanceDataset #type: ignore
from seqential import SimilarityNetwork #type: ignore
from tqdm import tqdm

distances_file = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\distances.npy' 
labels_file = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\labels.npy'    

dataset = DistanceDataset(distances_file, labels_file)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
similarity_model = SimilarityNetwork().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(similarity_model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    similarity_model.train()
    epoch_loss = 0.0
    with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = similarity_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
          
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': epoch_loss / (pbar.n or 1)})
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(similarity_model.state_dict(), r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\similarity_model_weights_2.pth')
print('Similarity network trained and saved.')
