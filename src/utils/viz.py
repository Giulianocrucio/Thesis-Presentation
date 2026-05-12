import matplotlib.pyplot as plt

def plot_training_curves(train_metric_history, train_loss_history, val_metric_history, val_loss_history, 
                         epochs, eval_every_n_epochs, save_path, is_class, model_name, dataset_name, test_metric, test_loss):
    plt.figure(figsize=(12, 7))
    plt.suptitle(f"Training Metrics: {model_name.upper()} on {dataset_name.upper()} Dataset", fontsize=16, fontweight='bold')
    
    epochs_range = list(range(1, epochs + 1))
    val_epochs = [e for e in epochs_range if e % eval_every_n_epochs == 0]
    
    metric1_name = 'Accuracy' if is_class else 'MAE'
    metric2_name = 'Loss' if is_class else 'MSE'

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_metric_history, label=f'Train {metric1_name}', alpha=0.8, linewidth=2)
    plt.plot(val_epochs, val_metric_history, label=f'Val {metric1_name}', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel(metric1_name)
    plt.title(f'{metric1_name} over Epochs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label=f'Train {metric2_name}', alpha=0.8, linewidth=2)
    plt.plot(val_epochs, val_loss_history, label=f'Val {metric2_name}', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel(metric2_name)
    plt.title(f'{metric2_name} over Epochs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Add test results and model info in the corner of the plot
    if is_class:
        stats_text = (
            f"Test Acc: {test_metric:.4f}\n"
            f"Test Loss: {test_loss:.4f}"
        )
    else:
        stats_text = (
            f"Test MAE: {test_metric:.4f}\n"
            f"Test MSE: {test_loss:.4f}"
        )
    plt.figtext(0.98, 0.02, stats_text, ha="right", fontsize=10, 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor='gray'))
    
    plt.savefig(save_path, dpi=300)
    plt.close()
