import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from .classifier_head import GraphClassifierHead

class GCN_ZINC(nn.Module):
    def __init__(self, num_node_classes=21, hidden_dim=145, num_layers=4, out_dim=1, node_dim=7, cfg=None):
        """
        GCN architecture optimized for the ZINC dataset (100k parameter budget).
        Reference: Dwivedi et al., "Benchmarking Graph Neural Networks" (2020/2023)
        """
        super(GCN_ZINC, self).__init__()
        
        self.dataset_name = cfg.data.name if cfg is not None else "zinc"
        if self.dataset_name == "zinc":
            self.node_emb = nn.Embedding(num_node_classes, hidden_dim)
        else:
            # For MUTAG and others, we use a linear layer on continuous node features
            self.node_emb = nn.Linear(node_dim, hidden_dim)
            
        # 2. GCN Message Passing Layers with Batch Norm
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # 1. OPTIMIZATION: Set normalize=False. We will handle it manually.
            self.convs.append(GCNConv(hidden_dim, hidden_dim, normalize=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # 3. MLP Head (Graph-level Readout to single scalar output)
        self.is_classification = cfg.data.get('classification_task', False) if cfg is not None else False
        
        if self.is_classification:
            dropout = cfg.model.get('dropout', 0.5) if cfg is not None else 0.5
            self.mlp = GraphClassifierHead(hidden_dim, out_dim, dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object containing x, edge_index, batch
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 2. SAFETY: use squeeze(-1) instead of squeeze()
        if self.dataset_name == "zinc":
            h = self.node_emb(x.long().squeeze(-1)) 
        else:
            h = self.node_emb(x)
            
        # 3. OPTIMIZATION: Precompute the GCN normalization once per batch
        edge_index, edge_weight = gcn_norm(
            edge_index, 
            num_nodes=h.size(0), 
            add_self_loops=True, 
            dtype=h.dtype
        )

        for conv, bn in zip(self.convs, self.batch_norms):
            h_in = h
            # 4. Pass the precomputed edge_weight into the convolution
            h = conv(h, edge_index, edge_weight)
            h = bn(h)
            # 5. OPTIMIZATION: inplace ReLU
            h = F.relu(h, inplace=True)
            h = h + h_in 
            
        # Readout: Global Mean Pooling
        h_graph = global_mean_pool(h, batch_idx)
        out = self.mlp(h_graph)
        
        if self.is_classification:
            return out
        else:
            return out.view(-1, 1)