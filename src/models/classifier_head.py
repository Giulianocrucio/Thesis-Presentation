import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphClassifierHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        # 1. Reduce dimensionality
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 2. Regularization to prevent overfitting on graph structures
        self.dropout = nn.Dropout(dropout)
        # 3. Output layer (logits)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, graph_embedding):
        # graph_embedding is from readout (e.g., global_mean_pool)
        x = F.relu(self.fc1(graph_embedding))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
