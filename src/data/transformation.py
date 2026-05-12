import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import slg2lib 

class SLG2Data(Data):
    """
    Custom PyG Data class to handle batching for the Super Line Graph.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l2_edge_index':
            # l2_edge_index points to L2 nodes. 
            # We increment it by the number of L2 nodes in the current graph.
            # l2_node_mapping shape is (2, num_l2_nodes), so size(1) gives the count.
            return self.l2_node_mapping.size(1) 

        if key == 'l2_node_mapping':
            # l2_node_mapping points to the original undirected edges.
            # We increment it by the number of unique undirected edges in the current graph.
            return self.undirected_edge_mask.sum().item()
            
        # Fallback to default PyG behavior for standard attributes like edge_index
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Tell PyG to concatenate these custom edge/mapping tensors along dimension 1 (columns)
        if key == 'l2_edge_index' or key == 'l2_node_mapping':
            return 1
            
        # For everything else (like undirected_edge_mask which is 1D), use default behavior (dim=0)
        return super().__cat_dim__(key, value, *args, **kwargs)

# Register the custom class as safe for PyTorch to load from disk
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([SLG2Data])


class L2Transform(BaseTransform):
    def forward(self, data):
        # 1 & 2. Pass PyG tensor directly to C++ and receive tensors back!
        l2_node_mapping, l2_edge_index = slg2lib.slg2_pg(data.edge_index)
        
        # 3. Make E_L2 undirected for PyG message passing
        if l2_edge_index.size(1) > 0:
            l2_edge_index = torch.cat([l2_edge_index, l2_edge_index.flip([0])], dim=1)
            
        # 4. The Undirected Edge Mask
        u, v = data.edge_index
        undirected_edge_mask = u < v
            
        # 5. Pack everything into our custom SLG2Data class
        slg2_data = SLG2Data()
        
        for key, item in data:
            slg2_data[key] = item
            
        slg2_data.l2_node_mapping = l2_node_mapping
        slg2_data.l2_edge_index = l2_edge_index
        slg2_data.undirected_edge_mask = undirected_edge_mask
            
        return slg2_data