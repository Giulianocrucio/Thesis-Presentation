import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from omegaconf import DictConfig
from .classifier_head import GraphClassifierHead

class MinMaxMeanPooling(nn.Module):
    """
    Computes min, max, and mean pooling in parallel using torch_scatter,
    and concatenates the results along the last dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, batch, dim_size=None):
        """
        Args:
            x (torch.Tensor): Node features of shape (num_nodes, hidden_dim)
            batch (torch.Tensor): Batch assignment for each node of shape (num_nodes,)
            dim_size (int, optional): The number of graphs in the batch.

        Returns:
            torch.Tensor: Concatenated pooled features of shape (batch_size, hidden_dim * 3)
        """
        # Fallback if dim_size is not explicitly provided
        if dim_size is None:
            dim_size = int(batch.max().item() + 1) if batch.numel() > 0 else 0

        # Mean pooling
        mean_pool = scatter(
            src=x,
            index=batch,
            dim=0,
            dim_size=dim_size,
            reduce='mean'
        )

        # Max pooling
        max_pool = scatter(
            src=x,
            index=batch,
            dim=0,
            dim_size=dim_size,
            reduce='max'
        )

        # Min pooling
        min_pool = scatter(
            src=x,
            index=batch,
            dim=0,
            dim_size=dim_size,
            reduce='min'
        )

        # Concatenate all pooled results (shape becomes hidden_dim * 3)
        return torch.cat([min_pool, max_pool, mean_pool], dim=-1)

class slg_v1(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, cfg: DictConfig):
        """
        Super Line Graph of Index 2 (SLG2) Model.
        
        Args:
            node_dim: Input dimension of node features (X)
            edge_dim: Input dimension of edge features (E)
            out_dim: Output dimension for final prediction (Y)
            cfg: Configuration dictionary
        ----------------------------------------------------
        We will use this parameter from cfg.model.:
            hidden_dim: Shared hidden representation dimension
            num_heads: Number of attention heads for the Set Transformer
        """
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        hidden_dim = cfg.model.hidden_dim
        num_heads = cfg.model.num_heads
        dropout = cfg.model.dropout
        num_layers_transformer = cfg.model.num_layers_transformer
        num_layers_ffn = cfg.model.num_layers_ffn
        # to make work if this parameter is not present in the config
        self.new_readout = cfg.model.get("new_readout", False)
        
        # print every cfg.model parameters
        print("-"*10)
        print("specs model")
        for key, value in cfg.model.items():
            print(f'{key}: {value}')
        print("-"*10)
        
        ############
        # ZINC case
        ############
        # if the dataset is ZINC we need an embedding for 
        # atom type (node feature) and edges features

        # We use nn.Embedding for boths

        # Unique node features: 21 (0 to 20)
        # Unique edge features: 3 (1,2,3)

        # Create embeddings for node and edge features
        self.use_edge_attr = edge_dim > 0

        self.dataset_name = cfg.data.name

        if self.dataset_name == "zinc":
            self.node_embedding = nn.Embedding(28, hidden_dim)
            self.edge_embedding = nn.Embedding(5, hidden_dim)
        else:
            if self.use_edge_attr:
                self.lin_edge = nn.Linear(edge_dim, hidden_dim)

        
        # H(0) - GCN(X,A)
        # Determine the actual input dimension to the GCN
        if self.dataset_name == "zinc":
            gcn_in_dim = hidden_dim  # Because x gets embedded to hidden_dim in the forward pass
        else:
            gcn_in_dim = node_dim    # Because x keeps its original node_dim

        # H(0) - GCN(X,A)
        self.gcn1 = GCNConv(gcn_in_dim, hidden_dim)

        # H(1) Set Transformer (Self-Attention over the {u, v, s, t} set)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            batch_first=True
        )
        self.set_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers_transformer
        )
        
        # H(2) GCN Layer for the Super Line Graph L2(G)
        self.gcn_l2 = GCNConv(hidden_dim, hidden_dim)

        # Readout layer with min-max-mean pooling
        self.multi_pool = MinMaxMeanPooling()
        
        # new_readout
        self.is_classification = cfg.data.get('classification_task', False)
        
        if self.new_readout:
            # Batch Norms for H(1) and H(2)
            self.bn1 = nn.LayerNorm(hidden_dim)
            self.bn2 = nn.LayerNorm(hidden_dim)
            # MLP Head (Graph-level Readout to single scalar output)
            if self.is_classification:
                self.ffn = GraphClassifierHead(hidden_dim, out_dim, dropout)
            else:
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, out_dim)
                )
        else:
            if self.is_classification:
                self.ffn = GraphClassifierHead(hidden_dim * 9, out_dim, dropout)
            else:
                # Final Feed Forward Neural Network with num_layers_ffn
                ffn_layers = []
                ffn_layers.append(nn.Linear(hidden_dim * 9, hidden_dim))
                ffn_layers.append(nn.LayerNorm(hidden_dim))
                ffn_layers.append(nn.GELU())
                ffn_layers.append(nn.Dropout(dropout))
                
                for _ in range(num_layers_ffn - 1):
                    ffn_layers.append(nn.Linear(hidden_dim, hidden_dim))
                    ffn_layers.append(nn.LayerNorm(hidden_dim))
                    ffn_layers.append(nn.GELU())
                    ffn_layers.append(nn.Dropout(dropout))
                ffn_layers.append(nn.Linear(hidden_dim, out_dim))
                self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        num_graphs = data.num_graphs

        # embeddings
        if self.dataset_name == "zinc":
            x = self.node_embedding(x.long().squeeze(-1))
            H0_edge = self.edge_embedding(edge_attr.long().squeeze(-1))
        elif self.use_edge_attr:
            H0_edge = self.lin_edge(edge_attr)


        # ==========================================
        # Compute H(0) - GCN(X,A)
        # ==========================================
        H0 = self.gcn1(x, edge_index)

        # ==========================================
        # Compute H(1)
        # ==========================================
        num_l2 = data.l2_node_mapping.shape[1] if hasattr(data, 'l2_node_mapping') else 0
        
        if num_l2 > 0:
            unique_edges = edge_index[:, data.undirected_edge_mask]
            
            e_idx = data.l2_node_mapping[0]
            f_idx = data.l2_node_mapping[1]
            
            u, v = unique_edges[0, e_idx], unique_edges[1, e_idx]
            s, t = unique_edges[0, f_idx], unique_edges[1, f_idx]
            
            # Gather features. Shape: [num_l2, hidden_dim]
            H0_u, H0_v = H0[u], H0[v]
            H0_s, H0_t = H0[s], H0[t]

            # Gether edge features
            H0_edge_e = H0_edge[e_idx]
            H0_edge_f = H0_edge[f_idx]
            
            # Stack into sets. Shape: [num_l2, 6, hidden_dim]
            H0_set = torch.stack([H0_u, H0_v, H0_s, H0_t, H0_edge_e, H0_edge_f], dim=1)
            
            # Set Transformer
            H1_seq = self.set_transformer(H0_set)

            # Mean Pooling over sequence - semplification of using PMA
            H1 = H1_seq.mean(dim=1)

            if self.new_readout:
                H0_mapped = (H0_u + H0_v + H0_s + H0_t) / 4.0
                H1 = H0_mapped + F.relu(self.bn1(H1))
            
            # Batch routing
            l2_batch = batch[u]
        else:
            H1 = torch.empty((0, self.hidden_dim), device=x.device)
            l2_batch = torch.empty((0,), dtype=torch.long, device=x.device)

        # ==========================================
        # Compute H(2) - GCN(L2(G), H1)
        # ==========================================
        if num_l2 > 0 and data.l2_edge_index.size(1) > 0:
            H2 = self.gcn_l2(H1, data.l2_edge_index)

            if self.new_readout:
                H2 = H1 + F.relu(self.bn2(H2))
            else:
                H2 = F.relu(H2)
        else:
            H2 = torch.empty((0, self.hidden_dim), device=x.device)

        # ==========================================
        # Pooling and Readout
        # ==========================================

        if self.new_readout: # get only the mean of H2
            H_pool = scatter(
                src=H2, 
                index=l2_batch, 
                dim=0, 
                dim_size=num_graphs, 
                reduce='mean'
            )
        else:
            H0_pool = self.multi_pool(H0, batch, dim_size=num_graphs)
            H1_pool = self.multi_pool(H1, l2_batch, dim_size=num_graphs)
            H2_pool = self.multi_pool(H2, l2_batch, dim_size=num_graphs)
            # H_pool will now be of shape [num_graphs, hidden_dim * 9]
            H_pool = torch.cat([H0_pool, H1_pool, H2_pool], dim=1)
        
        

        return self.ffn(H_pool)