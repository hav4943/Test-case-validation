import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
from network import BaseNetwork, SiameseNetwork  # type: ignore
from dataset import PairDataset  # type: ignore
from utils import ContrastiveLoss  # type: ignore
from sklearn.model_selection import train_test_split
from pad_collate import pad_collate_fn  # type: ignore
import torch.nn.functional as F
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Paths to your data directories
    pair_dir = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\Final_pair\pairs'
    label_dir = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\Final_pair\Labels'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using:', device)

    # Set seed
    seed = 36
    set_seed(seed)

    # Collect all pair files
    all_pair_files = sorted([
        f for f in os.listdir(pair_dir)
        if f.startswith('pair') and f.endswith('.npy')
    ])

    # Split data into training and validation sets
    train_files, val_files = train_test_split(all_pair_files, test_size=0.2, random_state=seed)

    # Verify no overlap
    overlap = set(train_files).intersection(set(val_files))
    if overlap:
        print(f"Overlap detected between training and validation sets: {overlap}")
    else:
        print("No overlap between training and validation sets.")

    # Prepare scaler using training data only
    train_data = []
    for pair_file in train_files:
        pair = np.load(os.path.join(pair_dir, pair_file), allow_pickle=True)
        scenario_1 = np.array(pair[0], dtype=np.float32)
        scenario_2 = np.array(pair[1], dtype=np.float32)
        train_data.append(scenario_1)
        train_data.append(scenario_2)
    train_data = np.concatenate(train_data, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_data)
    print('Scaler fitted successfully on training data!')

    # Save the scaler if needed
    joblib.dump(scaler, r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\New_SNN\scaler_1.pkl')

    # Create datasets using the split file lists
    train_dataset = PairDataset(pair_dir, label_dir, scaler, file_list=train_files)
    val_dataset = PairDataset(pair_dir, label_dir, scaler, file_list=val_files)

    # Verify dataset sizes
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Create DataLoaders
    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=pad_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=pad_collate_fn
    )
    
    input_size = train_dataset[0][0].shape[1]  # Feature size from the first scenario in the dataset

    base_network = BaseNetwork(input_size=input_size).to(device)
    model = SiameseNetwork(base_network).to(device)

    criterion = ContrastiveLoss(margin=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    num_epochs = 1000 # Adjust as needed

    # Initialize lists to store distances and labels

    train_losses = []  # To store training losses per epoch
    val_losses = []    # To store validation losses per epoch

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        total_train_samples = 0     
        distances = []
        labels = []
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for input1, input2, batch_labels, lengths1, lengths2 in train_dataloader:
                input1, input2 = input1.to(device), input2.to(device)
                batch_labels = batch_labels.float().to(device)
                lengths1 = lengths1.to(device)
                lengths2 = lengths2.to(device)

                optimizer.zero_grad()

                # Forward pass
                output1, output2 = model(input1, input2, lengths1, lengths2)

                # Compute loss
                loss = criterion(output1, output2, batch_labels)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                train_total_loss += loss.item() * input1.size(0)    
                total_train_samples += input1.size(0)
                # train_total_loss += loss.item()

                # Compute distance metric and collect distances and labels
                euclidean_distance = F.pairwise_distance(output1, output2)
                distances.extend(euclidean_distance.detach().cpu().numpy())
                labels.extend(batch_labels.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix({'Training Loss': train_total_loss / total_train_samples})

        avg_train_loss = train_total_loss / total_train_samples #len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Validation loop (optional)
        model.eval()
        val_total_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Validation {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for input1, input2, batch_labels, lengths1, lengths2 in val_dataloader:
                    input1, input2 = input1.to(device), input2.to(device)
                    batch_labels = batch_labels.float().to(device)
                    lengths1 = lengths1.to(device)
                    lengths2 = lengths2.to(device)

                    output1, output2 = model(input1, input2, lengths1, lengths2)
                    loss = criterion(output1, output2, batch_labels)
                    val_total_loss += loss.item() * input1.size(0)
                    total_val_samples += input1.size(0)
                    # val_total_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({'Validation Loss': val_total_loss / total_val_samples})
            avg_val_loss = val_total_loss / total_val_samples
            val_losses.append(avg_val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    # After training, save the distances and labels
    distances = np.array(distances)
    labels = np.array(labels)

    np.save(r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\distances_1.npy', distances)
    np.save(r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\labels_1.npy', labels)
    print('Distances and labels saved.')

    # Save the model
    save_path = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\siamese_model_weights.pth'  # Replace with your actual path
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    print('Training finished')
     # Plot training and validation losses
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', color = 'r')
    plt.plot(epochs, val_losses, label='Validation Loss', color = 'b')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')  # Save the figure
    plt.show()