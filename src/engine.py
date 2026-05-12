import torch
import torch.nn.functional as F
from data.prep import prepare_batch
from utils.io import setup_logger

logger = setup_logger(__name__)

def train_epoch(model, loader, optimizer, criterion, device, cfg, epoch):
    model.train()
    total_l1_or_acc = 0
    total_l2_or_loss = 0
    num_samples = 0
    is_class = cfg.data.get('classification_task', False)
    
    for step, batch in enumerate(loader):
        batch = prepare_batch(batch, device, is_class)
        batch_size = batch.y.size(0)
        
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        
        # Gradient clipping is highly recommended for Transformer/GNN layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            if is_class:
                preds = out.argmax(dim=1)
                metric1 = (preds == batch.y).sum().item() # accuracy count
                metric2 = loss.item() * batch_size  # total loss
            else:
                metric1 = F.l1_loss(out, batch.y, reduction='sum').item()
                metric2 = F.mse_loss(out, batch.y, reduction='sum').item()

        total_l1_or_acc += metric1
        total_l2_or_loss += metric2
        num_samples += batch_size
        
        # Log every N steps
        if (step + 1) % cfg.training.log_every_n_steps == 0:
            logger.info(f"Epoch [{epoch:03d}] Step [{step + 1:03d}/{len(loader)}] "
                        f"- Batch Train Loss: {loss.item():.4f}")

    return {"metric": total_l1_or_acc / num_samples, "loss": total_l2_or_loss / num_samples}

@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    total_l1_or_acc = 0
    total_l2_or_loss = 0
    num_samples = 0
    is_class = cfg.data.get('classification_task', False)
    
    for batch in loader:
        batch = prepare_batch(batch, device, is_class)
        batch_size = batch.y.size(0)
        
        out = model(batch)
        loss = criterion(out, batch.y)
        
        if is_class:
            preds = out.argmax(dim=1)
            metric1 = (preds == batch.y).sum().item()
            metric2 = loss.item() * batch_size
        else:
            metric1 = F.l1_loss(out, batch.y, reduction='sum').item()
            metric2 = F.mse_loss(out, batch.y, reduction='sum').item()
            
        total_l1_or_acc += metric1
        total_l2_or_loss += metric2
        num_samples += batch_size
        
    return {"metric": total_l1_or_acc / num_samples, "loss": total_l2_or_loss / num_samples}
