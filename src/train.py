import os
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime

# Import modules
from data import load_zinc_benchmark, load_mutag_benchmark, prepare_batch, load_NCI1_benchmark, load_ENZYMES_benchmark
from models import build_model
from engine import train_epoch, evaluate
from utils.io import setup_logger, CSVLogger
from utils.metrics import get_loss_fn
from utils.viz import plot_training_curves

logger = setup_logger(__name__)

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------")
    
    # ---------------------------------------------------------
    # GraphGym Dispatcher
    # ---------------------------------------------------------
    if cfg.system.get('use_graphgym', False):
        logger.info("Running with GraphGym Pipeline")
        from utils.graphgym_mapper import setup_graphgym_cfg
        import torch_geometric.graphgym.register as register
        from torch_geometric.graphgym.loader import create_loader
        from torch_geometric.graphgym.model_builder import create_model
        from torch_geometric.graphgym.train import train
        
        # Map Hydra config to GraphGym
        gg_cfg = setup_graphgym_cfg(cfg)
        
        # Set seeds for GraphGym
        from torch_geometric.graphgym.utils.seed import set_seed
        set_seed(gg_cfg.seed)
        
        loaders = create_loader()
        model = create_model()
        logger.info(model)
        
        train(model, loaders, optimizer_config=gg_cfg.optim) # Depending on PyG version, this might differ slightly, using standard GraphGym train
        
        logger.info("GraphGym pipeline finished.")
        return
    # ---------------------------------------------------------

    
    # Set seed for reproducibility
    torch.manual_seed(cfg.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.system.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create timestamped log directory: year_month_day_hour_min_sec
    date_hours = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs", date_hours)
    os.makedirs(log_dir, exist_ok=True)
    
    config_save_path = os.path.join(log_dir, "config.yaml")
    OmegaConf.save(config=cfg, f=config_save_path)
    
    csv_path = os.path.join(log_dir, "results.csv")
    csv_logger = CSVLogger(csv_path, is_classification=cfg.data.get('classification_task', False))

    checkpoint_path = os.path.join(log_dir, "best_model.pt")

    # 1. Load Data
    logger.info(f"Loading {cfg.data.name} datasets...")
    if cfg.data.name == 'zinc':
        train_loader, val_loader, test_loader = load_zinc_benchmark(
            batch_size=cfg.training.batch_size,
            use_l2=cfg.system.use_l2,
            num_workers=cfg.system.num_workers
        )
    elif cfg.data.name == 'mutag':
        train_loader, val_loader, test_loader = load_mutag_benchmark(
            batch_size=cfg.training.batch_size,
            use_l2=cfg.system.use_l2,
            num_workers=cfg.system.num_workers,
            seed=cfg.system.seed
        )
    elif cfg.data.name == 'nci1':
        train_loader, val_loader, test_loader = load_NCI1_benchmark(
            batch_size=cfg.training.batch_size,
            use_l2=cfg.system.use_l2,
            num_workers=cfg.system.num_workers,
            seed=cfg.system.seed
        )
    elif cfg.data.name == 'enzymes':
        train_loader, val_loader, test_loader = load_ENZYMES_benchmark(
            batch_size=cfg.training.batch_size,
            use_l2=cfg.system.use_l2,
            num_workers=cfg.system.num_workers,
            seed=cfg.system.seed
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")

    if cfg.data.get('classification_task', False):
        from omegaconf import open_dict
        with open_dict(cfg):
            if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'num_classes'):
                cfg.data.num_classes = train_loader.dataset.dataset.num_classes
            elif hasattr(train_loader.dataset, 'num_classes'):
                cfg.data.num_classes = train_loader.dataset.num_classes

    # 2. Initialize Model
    logger.info("Initializing model...")
    init_batch = next(iter(train_loader))
    # Move batch to cpu for building model
    init_batch = prepare_batch(init_batch, torch.device('cpu'), is_classification=cfg.data.get('classification_task', False))
    logger.info(f"Sample data batch tensor device: {init_batch.x.device}")
    
    model = build_model(cfg, init_batch)
    model = model.to(device)
    logger.info(f"Model initialized and placed on device: {next(model.parameters()).device}")
    
    # Calculate parameters and memory footprint
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    logger.info(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    logger.info(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    # 3. Setup Optimizer, Loss, and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )
    # ZINC standard practice: reduce LR when validation plateaus
    eval_every_n = cfg.training.get('eval_every_n_epochs', 1)
    adjusted_patience = max(1, cfg.training.patience // eval_every_n)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=adjusted_patience, min_lr=1e-5
    )
    criterion = get_loss_fn(cfg)

    # 4. Training Loop
    logger.info("=== Starting Training ===")
    best_val_loss = float('inf')
    last_val_loss = None
    
    train_metric_history = []
    train_loss_history = []
    val_metric_history = []
    val_loss_history = []
    
    start_time = time.time()
    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, cfg, epoch)
        train_metric, train_loss = train_res["metric"], train_res["loss"]
        train_metric_history.append(train_metric)
        train_loss_history.append(train_loss)
        
        val_metric, val_loss_val = None, None
        
        # Validation & Logging
        if epoch % eval_every_n == 0:
            val_res = evaluate(model, val_loader, criterion, device, cfg)
            val_metric, val_loss_val = val_res["metric"], val_res["loss"]
            val_metric_history.append(val_metric)
            val_loss_history.append(val_loss_val)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            if cfg.data.get('classification_task', False):
                last_val_loss = val_loss_val # For classification, val_mse is the loss
                logger.info(f"Epoch [{epoch:03d}/{cfg.training.epochs}] - "
                            f"Train Acc: {train_metric:.4f} - Val Acc: {val_metric:.4f} - Val Loss: {last_val_loss:.4f} - LR: {current_lr:.6f}")
            else:
                last_val_loss = val_metric if cfg.training.loss.lower() in ["l1", "mae"] else val_loss_val
                logger.info(f"Epoch [{epoch:03d}/{cfg.training.epochs}] - "
                            f"Train MAE: {train_metric:.4f} - Val MAE: {val_metric:.4f} - LR: {current_lr:.6f}")
                
            scheduler.step(last_val_loss)
            
            # Best Checkpoint saving
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': last_val_loss,
                }, checkpoint_path)
                logger.info(f"[*] New best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
                
        csv_logger.log_epoch(
            train_metric, train_loss, 
            val_metric if val_metric is not None else "", 
            val_loss_val if val_loss_val is not None else ""
        )
                
    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Training Complete in {total_time:.2f} minutes ===")

    # 5. Final Testing
    logger.info("=== Starting Final Evaluation on Test Set ===")
    # Load the best weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    else:
        logger.warning("No checkpoint found — testing with final model weights.")

    test_res = evaluate(model, test_loader, criterion, device, cfg)
    test_metric, test_loss = test_res["metric"], test_res["loss"]
    
    csv_logger.log_test(test_metric, test_loss)
    
    logger.info(f"--- TEST RESULTS ---")
    if cfg.data.get('classification_task', False):
        logger.info(f"Test Accuracy: {test_metric:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
    else:
        logger.info(f"Test MAE (L1): {test_metric:.4f}")
        logger.info(f"Test MSE (L2): {test_loss:.4f}")
    logger.info("====================")
    
    # 6. Plotting
    plot_path = os.path.join(log_dir, "plot_results.png")
    
    plot_training_curves(
        train_metric_history, train_loss_history, val_metric_history, val_loss_history,
        cfg.training.epochs, cfg.training.eval_every_n_epochs, plot_path, 
        cfg.data.get('classification_task', False), cfg.model.name, cfg.data.name, 
        test_metric, test_loss
    )
    logger.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()