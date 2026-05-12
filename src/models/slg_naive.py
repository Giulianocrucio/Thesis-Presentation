import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.nn import global_mean_pool, GlobalAttention
from omegaconf import DictConfig
from .classifier_head import GraphClassifierHead

class SimpleSetTransformer(nn.Module):
    """
    A lightweight Set Transformer block using PyTorch's native MultiheadAttention.
    Pools a set of vectors into a single permutation-invariant representation.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, set_size, hidden_dim]
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ffn(x)
        # Permutation invariant pooling (mean across the set size)
        return x.mean(dim=1)


class slg_naive(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, cfg: DictConfig):
        super().__init__()
        
        self.hidden_dim = cfg.model.get('hidden_dim', 64)
        self.dropout_rate = cfg.model.get('dropout', 0.0)
        
        self.dataset_name = cfg.data.name
        self.use_edge_attr = edge_dim > 0
        
        # H^(0) Node and Edge Projections
        if self.dataset_name == "zinc":
            self.node_embed = nn.Embedding(28, self.hidden_dim)
            self.edge_embed = nn.Embedding(5, self.hidden_dim)
            self.node_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.edge_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.node_proj = nn.Linear(node_dim, self.hidden_dim)
            if self.use_edge_attr:
                self.edge_proj = nn.Linear(edge_dim, self.hidden_dim)

        # Set Transformers
        num_heads_st1 = cfg.model.get('num_heads_st1', 4)
        num_heads_st2 = cfg.model.get('num_heads_st2', 4)
        self.set_transformer_1 = SimpleSetTransformer(self.hidden_dim, num_heads_st1)
        self.set_transformer_2 = SimpleSetTransformer(self.hidden_dim, num_heads_st2)
        
        # Configuration for specific twists
        self.topology_info = cfg.model.get('topology_info', True)
        self.use_in_out = cfg.model.get('use_in_out', False)
        
        # Determine the dimension of H^(1) after twists
        self.h1_dim = self.hidden_dim
        if self.topology_info:
            self.h1_dim += 1
        if self.use_in_out:
            self.h1_dim += 1
            
        # Readout Configuration
        self.readout_type = cfg.model.get('readout_type', 'mean')
        self.readout_dim = self.h1_dim
        
        if self.readout_type == 'attention':
            self.att_gate = nn.Sequential(
                nn.Linear(self.h1_dim, self.h1_dim // 2),
                nn.ReLU(),
                nn.Linear(self.h1_dim // 2, 1)
            )
            self.global_att_pool = GlobalAttention(gate_nn=self.att_gate)
        elif self.readout_type == 'set_transformer':
            num_heads_readout = cfg.model.get('num_heads_readout', 4)
            self.st_d_model = self.h1_dim
            if self.st_d_model % num_heads_readout != 0:
                self.st_d_model += num_heads_readout - (self.st_d_model % num_heads_readout)
            self.readout_dim = self.st_d_model
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.st_d_model, 
                nhead=num_heads_readout, 
                batch_first=True
            )
            self.readout_st = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification Head / MLP
        self.is_classification = cfg.data.get('classification_task', False)
        
        if self.is_classification:
            self.mlp = GraphClassifierHead(self.readout_dim, out_dim, self.dropout_rate)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.readout_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, out_dim)
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs

        # ==========================================
        # 1. Compute H^(0) and H_edges
        # ==========================================
        if self.dataset_name == "zinc":
            H0 = self.node_proj(self.node_embed(x.long().squeeze(-1)))
            H_edge = self.edge_proj(self.edge_embed(edge_attr.long().squeeze(-1)))
        else:
            H0 = self.node_proj(x.float())
            if self.use_edge_attr:
                H_edge = self.edge_proj(edge_attr.float())

        # Check if L2 mapping exists to prevent breaking on graphs with no edges
        num_l2 = data.l2_node_mapping.size(1) if hasattr(data, 'l2_node_mapping') else 0
        
        if num_l2 > 0:
            # unique_edges strips out reverse directions if graph was made undirected
            unique_edges = edge_index[:, data.undirected_edge_mask] if hasattr(data, 'undirected_edge_mask') else edge_index
            
            # Extract original edge indices for e and f
            e_idx = data.l2_node_mapping[0]
            f_idx = data.l2_node_mapping[1]
            
            # Extract nodes u1, u2 for edge e, and v1, v2 for edge f
            u1, u2 = unique_edges[0, e_idx], unique_edges[1, e_idx]
            v1, v2 = unique_edges[0, f_idx], unique_edges[1, f_idx]

            # Fetch features mapping to the nodes and edges
            h0_u1, h0_u2 = H0[u1], H0[u2]
            h0_v1, h0_v2 = H0[v1], H0[v2]
            
            has_edge_attr = self.dataset_name == "zinc" or self.use_edge_attr
            if has_edge_attr:
                # Need to use undirected_edge_mask for edges too if we map them
                H_edge_masked = H_edge[data.undirected_edge_mask] if hasattr(data, 'undirected_edge_mask') else H_edge
                h_e = H_edge_masked[e_idx]
                h_f = H_edge_masked[f_idx]

            # ==========================================
            # 2. Compute h^(1)_e and h^(1)_f
            # ==========================================
            # Stack into sets of size 2 -> Shape: [num_l2_nodes, 2, hidden_dim]
            set_e = torch.stack([h0_u1, h0_u2], dim=1)
            set_f = torch.stack([h0_v1, h0_v2], dim=1)

            if has_edge_attr:
                h1_e = self.set_transformer_1(set_e) + h_e
                h1_f = self.set_transformer_1(set_f) + h_f
            else:
                h1_e = self.set_transformer_1(set_e)
                h1_f = self.set_transformer_1(set_f)

            # ==========================================
            # 3. Compute h^(1)_{e,f}
            # ==========================================
            set_ef = torch.stack([h1_e, h1_f], dim=1)
            mean_nodes = (h0_u1 + h0_u2 + h0_v1 + h0_v2) / 4.0
            
            H1 = self.set_transformer_2(set_ef) + mean_nodes

            # ==========================================
            # 4. Twist 1: Topology Info (Incidence)
            # ==========================================
            if self.topology_info:
                # 1 if incident (share a node), 0 otherwise
                is_incident = ((u1 == v1) | (u1 == v2) | (u2 == v1) | (u2 == v2)).float().unsqueeze(-1)
                H1 = torch.cat([H1, is_incident], dim=-1)

            # ==========================================
            # 5. Twist 2: L2 Degree Info
            # ==========================================
            if self.use_in_out:
                if hasattr(data, 'l2_edge_index') and data.l2_edge_index.size(1) > 0:
                    l2_deg = degree(data.l2_edge_index[0], num_nodes=num_l2).float().unsqueeze(-1)
                else:
                    l2_deg = torch.zeros((num_l2, 1), device=H1.device)
                H1 = torch.cat([H1, l2_deg], dim=-1)

            # Assign batch index for L2 nodes (maps to the graph they belong to)
            l2_batch = batch[u1]
            
        else:
            # Fallback for empty/disconnected graphs
            H1 = torch.empty((0, self.h1_dim), device=x.device)
            l2_batch = torch.empty((0,), dtype=torch.long, device=x.device)

        # ==========================================
        # 6. Readout Stage
        # ==========================================
        if H1.size(0) == 0:
            pooled = torch.zeros((num_graphs, self.readout_dim), device=x.device)
        else:
            if self.readout_type == 'mean':
                pooled = global_mean_pool(H1, l2_batch)
                
            elif self.readout_type == 'attention':
                pooled = self.global_att_pool(H1, l2_batch)
                
            elif self.readout_type == 'set_transformer':
                if self.st_d_model > self.h1_dim:
                    H1_padded = F.pad(H1, (0, self.st_d_model - self.h1_dim))
                else:
                    H1_padded = H1
                # Pad batch to dense representation for standard Transformer Encoder
                dense_H1, padding_mask = to_dense_batch(H1_padded, l2_batch) 
                
                # TransformerEncoder expects padding_mask where True = ignore
                st_out = self.readout_st(dense_H1, src_key_padding_mask=~padding_mask)
                
                # Re-aggregate correctly using mask
                mask_expanded = padding_mask.unsqueeze(-1).float()
                pooled = (st_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown readout type: {self.readout_type}")

        if pooled.size(0) < num_graphs:
            full_pooled = torch.zeros((num_graphs, self.readout_dim), device=pooled.device)
            unique_batches = torch.unique(l2_batch)
            full_pooled[unique_batches] = pooled
            pooled = full_pooled

        # ==========================================
        # 7. Classification / MLP Head
        # ==========================================
        out = self.mlp(pooled)
        return out