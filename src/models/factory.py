from .mod_slg2 import mod_slg2
from .slg_v1 import slg_v1
from .gcn import GCN_ZINC
from .mod_slg2_v2 import mod_slg2_v2
from .slg_naive import slg_naive
from .slg_advance import slg_advanced

def build_model(cfg, batch):
    """Dynamically builds the model based on the first batch dimensions."""
    node_dim = batch.x.size(1)
    edge_dim = batch.edge_attr.size(1) if getattr(batch, 'edge_attr', None) is not None else 0
    
    if cfg.data.get('classification_task', False):
        # We can dynamically get num_classes from dataset or use 2 for MUTAG
        out_dim = cfg.data.get('num_classes', 2)
    else:
        out_dim = batch.y.shape[1] if getattr(batch, 'y', None) is not None and batch.y.dim() > 1 else 1
    
    if cfg.model.name == 'mod_slg2':
        model = mod_slg2(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            cfg=cfg
        )
    elif cfg.model.name == 'slg_v1':
        model = slg_v1(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            cfg=cfg
        )
    elif cfg.model.name == 'gcn':
        model = GCN_ZINC(
            num_node_classes=cfg.data.get('num_node_features', 21), # Safely pull from config, default 21
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            out_dim=out_dim,
            node_dim=node_dim,
            cfg=cfg
        )
    elif cfg.model.name == 'mod_slg2_v2':
        model = mod_slg2_v2(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            cfg=cfg
        )
    elif cfg.model.name == 'slg_naive':
        from .slg_naive import slg_naive
        model = slg_naive(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            cfg=cfg
        )
    elif cfg.model.name == 'slg_advanced':
        from .slg_advance import slg_advanced
        model = slg_advanced(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            cfg=cfg
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    
    return model
