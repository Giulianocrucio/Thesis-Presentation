import torch.nn as nn

def get_loss_fn(cfg):
    """Returns the primary loss function."""
    if cfg.data.get('classification_task', False):
        return nn.CrossEntropyLoss()
        
    loss_name = cfg.training.loss
    if loss_name.lower() in ["l1", "mae"]:
        return nn.L1Loss()
    elif loss_name.lower() in ["l2", "mse"]:
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss {loss_name}")
