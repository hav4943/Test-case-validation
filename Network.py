import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, lstm_layers=1, dropout=0.5):
        super(BaseNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.relu3 = nn.ReLU()

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=False, dropout=dropout)

        # Attention Layer
        self.attention = nn.Linear(hidden_size , 1)  # 2x for bidirectional LSTM
        self.softmax = nn.Softmax(dim=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size , 128)  # *2 for bidirectional
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)

        # Convolutional layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)

        # Pack the padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)

        attn_weights = self.attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_len)
        attn_weights[~mask] = float('-inf')
        attn_weights = self.softmax(attn_weights)

        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)

        # Fully connected layers
        x = self.fc1(context_vector)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        return x

class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward_once(self, x, lengths):
        return self.base_network(x, lengths)

    def forward(self, input1, input2, lengths1, lengths2):
        output1 = self.forward_once(input1, lengths1)
        output2 = self.forward_once(input2, lengths2)
        return output1, output2
