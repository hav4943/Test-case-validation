import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length in each batch.
    """
    
    input1_list, input2_list, label_list, lengths1_list, lengths2_list = zip(*batch)

    input1_padded = pad_sequence(input1_list, batch_first=True)
    input2_padded = pad_sequence(input2_list, batch_first=True)

    labels = torch.stack(label_list)
    lengths1 = torch.stack(lengths1_list)
    lengths2 = torch.stack(lengths2_list)

    return input1_padded, input2_padded, labels, lengths1, lengths2
