import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from omegaconf import DictConfig
from .classifier_head import GraphClassifierHead

class FFN(nn.Module):
    """
    A simple feedforward neural network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, num_layers: int, use_batchnorm=True, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_batchnorm = use_batchnorm
        self.dropout = nn.Dropout(dropout)
        dims = [input_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        self.batchnorms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batchnorm and i < num_layers - 1:
                self.batchnorms.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batchnorm and i < len(self.layers) - 1:
                x = self.batchnorms[i](x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        return x


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

class mod_slg2(nn.Module):
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

        # H(1) Feed Forward Neural Networks
        self.hidden_dim_1 = cfg.model.get('hidden_dim_1', 64)
        self.num_layers1 = cfg.model.get('num_layers1', 2)
        self.edge_node_hidden_dim = cfg.model.get('edge_node_hidden_dim', 64)
        self.n_layers_ff_ef = cfg.model.get('n_layers_ff_ef', 2)

        # ff_ne input depends on whether edge attributes are used
        ff_ne_in_dim = hidden_dim * 3 if (self.dataset_name == "zinc" or self.use_edge_attr) else hidden_dim * 2
        
        self.ff_ne = FFN(
            input_dim=ff_ne_in_dim, 
            hidden_dim=self.hidden_dim_1, 
            out_dim=self.edge_node_hidden_dim, 
            num_layers=self.num_layers1, 
            dropout=dropout
        )
        
        self.ff_ef = FFN(
            input_dim=self.edge_node_hidden_dim * 2,
            hidden_dim=self.edge_node_hidden_dim * 2,
            out_dim=hidden_dim,
            num_layers=self.n_layers_ff_ef,
            dropout=dropout
        )
        
        # H(2) GCN Layer for the Super Line Graph L2(G)
        self.gcn_l2 = GCNConv(hidden_dim, hidden_dim)

        # Readout layer with min-max-mean pooling
        self.multi_pool = MinMaxMeanPooling()
        
        # Final Feed Forward Neural Network with num_layers_ffn
        self.is_classification = cfg.data.get('classification_task', False)
        
        if self.is_classification:
            self.ffn = GraphClassifierHead(hidden_dim * 9, out_dim, dropout)
        else:
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
            
            # --- OPTIMIZATION: Compute h_edge for all unique edges once ---
            u_all, v_all = unique_edges[0], unique_edges[1]
            H0_u_all, H0_v_all = H0[u_all], H0[v_all]

            has_edge_attr = self.dataset_name == "zinc" or self.use_edge_attr
            if has_edge_attr:
                # Gather edge features for unique edges using the mask
                H0_edge_all = H0_edge[data.undirected_edge_mask]
                
                concat_1_all = torch.cat([H0_u_all, H0_edge_all, H0_v_all], dim=-1)
                concat_2_all = torch.cat([H0_v_all, H0_edge_all, H0_u_all], dim=-1)
            else:
                concat_1_all = torch.cat([H0_u_all, H0_v_all], dim=-1)
                concat_2_all = torch.cat([H0_v_all, H0_u_all], dim=-1)
                
            h_edge_all = 0.5 * (self.ff_ne(concat_1_all) + self.ff_ne(concat_2_all))
            
            # Now gather the precomputed features for each pair
            h_e = h_edge_all[e_idx]
            h_f = h_edge_all[f_idx]
            
            concat_1_ef = torch.cat([h_f, h_e], dim=-1)
            concat_2_ef = torch.cat([h_e, h_f], dim=-1)
            
            H1 = 0.5 * (self.ff_ef(concat_1_ef) + self.ff_ef(concat_2_ef))
            
            # Batch routing
            u = u_all[e_idx]
            l2_batch = batch[u]
        else:
            H1 = torch.empty((0, self.hidden_dim), device=x.device)
            l2_batch = torch.empty((0,), dtype=torch.long, device=x.device)

        # ==========================================
        # Compute H(2) - GCN(L2(G), H1)
        # ==========================================
        if num_l2 > 0 and data.l2_edge_index.size(1) > 0:
            H2 = self.gcn_l2(H1, data.l2_edge_index)
            H2 = F.relu(H2)
        else:
            H2 = torch.empty((0, self.hidden_dim), device=x.device)

        # ==========================================
        # Pooling and Readout
        # ==========================================
        H0_pool = self.multi_pool(H0, batch, dim_size=num_graphs)
        H1_pool = self.multi_pool(H1, l2_batch, dim_size=num_graphs)
        H2_pool = self.multi_pool(H2, l2_batch, dim_size=num_graphs)

        # H_pool will now be of shape [num_graphs, hidden_dim * 9]
        H_pool = torch.cat([H0_pool, H1_pool, H2_pool], dim=1)

        return self.ffn(H_pool)