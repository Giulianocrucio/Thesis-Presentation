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

class mod_slg2_v2(nn.Module):
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
        self.only_last_h = cfg.model.get('only_last_h', False)
        self.include_topological = cfg.model.get('include_topological', False)
        
        
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
        h1_dim = hidden_dim + 1 if self.include_topological else hidden_dim
        
        self.gcn_l2 = GCNConv(h1_dim, h1_dim)

        # Batch Norms for H(1) and H(2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(h1_dim)
        
        # Final Feed Forward Neural Network with num_layers_ffn
        self.is_classification = cfg.data.get('classification_task', False)
        
        # We concatenate mean of H(1)_new and H(2)_new, so input is h1_dim * 2
        if cfg.model.only_last_h:
            last_dim = h1_dim
        else:
            last_dim = h1_dim * 2
            
        if self.is_classification:
            self.ffn = GraphClassifierHead(last_dim, out_dim, dropout)
        else:
            ffn_layers = []
            ffn_layers.append(nn.Linear(last_dim, hidden_dim))
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
            
            # H(0) and H(1) have different shapes (nodes vs L2 nodes). 
            # We map H0 to L2 nodes by taking the mean of the involved nodes.
            u = u_all[e_idx]
            v = v_all[e_idx]
            s = u_all[f_idx]
            t = v_all[f_idx]
            H0_u, H0_v = H0[u], H0[v]
            H0_s, H0_t = H0[s], H0[t]
            H0_mapped = (H0_u + H0_v + H0_s + H0_t) / 4.0
            H1_new = H0_mapped + F.relu(self.bn1(H1))
            
            if self.include_topological:
                is_incident = ((u == s) | (u == t) | (v == s) | (v == t)).float().unsqueeze(1)
                H1_new = torch.cat([H1_new, is_incident], dim=1)
            
            # Batch routing
            l2_batch = batch[u]
        else:
            h1_dim = self.hidden_dim + 1 if self.include_topological else self.hidden_dim
            H1_new = torch.empty((0, h1_dim), device=x.device)
            l2_batch = torch.empty((0,), dtype=torch.long, device=x.device)

        # ==========================================
        # Compute H(2) - GCN(L2(G), H1_new)
        # ==========================================
        if num_l2 > 0:
            if data.l2_edge_index.size(1) > 0:
                l2_edges = data.l2_edge_index
            else:
                l2_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
            H2 = self.gcn_l2(H1_new, l2_edges)
            H2_new = H1_new + F.relu(self.bn2(H2))
        else:
            h1_dim = self.hidden_dim + 1 if self.include_topological else self.hidden_dim
            H2_new = torch.empty((0, h1_dim), device=x.device)

        # ==========================================
        # Pooling and Readout
        # ==========================================
        if num_l2 > 0:
            H2_pool = scatter(H2_new, l2_batch, dim=0, dim_size=num_graphs, reduce='mean')
            if not self.only_last_h: 
                H1_pool = scatter(H1_new, l2_batch, dim=0, dim_size=num_graphs, reduce='mean')
        else:
            h1_dim = self.hidden_dim + 1 if self.include_topological else self.hidden_dim
            H2_pool = torch.zeros((num_graphs, h1_dim), device=x.device)
            if not self.only_last_h:
                H1_pool = torch.zeros((num_graphs, h1_dim), device=x.device)
        
        if self.only_last_h:
            H_pool = H2_pool
        else:
            # H_pool will now be of shape [num_graphs, hidden_dim * 2]
            H_pool = torch.cat([H1_pool, H2_pool], dim=1)

        return self.ffn(H_pool)