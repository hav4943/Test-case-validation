import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos_weight = 1.0  # Weight for similar pairs
        neg_weight = 1.0  # Weight for dissimilar pairs (adjust as needed)

        loss_contrastive = torch.mean(
            pos_weight * (label) * torch.pow(euclidean_distance, 2) +  # For similar pairs (label = 1)
            neg_weight * (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # For dissimilar pairs (label = 0)
        )

        return loss_contrastive
