from torch_geometric.graphgym.config import cfg as gg_cfg
from omegaconf import DictConfig

def setup_graphgym_cfg(hydra_cfg: DictConfig):
    """
    Maps a Hydra DictConfig to the GraphGym YACS configuration object.
    This acts as the bridge between your existing configuration setup
    and the GraphGym pipeline.
    """
    # 1. Dataset Configuration
    gg_cfg.dataset.name = hydra_cfg.data.name
    # Handle specific dataset format differences here if needed
    if hydra_cfg.data.name.lower() in ['zinc', 'mutag', 'nci1', 'enzymes']:
        gg_cfg.dataset.format = 'PyG'
        
    # 2. Training Configuration
    gg_cfg.train.batch_size = hydra_cfg.training.batch_size
    gg_cfg.train.epochs = hydra_cfg.training.epochs
    gg_cfg.train.eval_period = hydra_cfg.training.eval_every_n_epochs
    
    # 3. Optimizer Configuration
    gg_cfg.optim.base_lr = hydra_cfg.training.lr
    gg_cfg.optim.weight_decay = hydra_cfg.training.weight_decay
    
    # 4. Model Configuration (Mapping your custom model name)
    # This assumes you have registered your model with @register_network('slg_advanced')
    gg_cfg.model.type = hydra_cfg.model.name 
    
    # 5. System Configuration
    gg_cfg.seed = hydra_cfg.system.seed
    gg_cfg.num_workers = hydra_cfg.system.num_workers
    
    # You can add more complex mappings here, 
    # like passing specific hidden dimensions or layers:
    # gg_cfg.gnn.dim_inner = hydra_cfg.model.hidden_dim
    
    return gg_cfg
