# models/online/visualize.py
"""
Visualization utilities for training and evaluation metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Optional


class TrainingVisualizer:
    """
    Visualizer for training and evaluation metrics.
    Generates plots for loss curves, CER/WER, and other metrics.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directory to save plots. If None, uses './experiments/figures'
        """
        if output_dir is None:
            output_dir = Path("./experiments/figures")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metrics
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_cer': [],
            'val_wer': [],
            'learning_rates': [],
            'train_cer': [],  # Optional
            'train_wer': []   # Optional
        }
    
    def update(self, epoch: int, train_loss: float, val_cer: float, val_wer: float, 
               learning_rate: float, train_cer: float = None, train_wer: float = None):
        """
        Update metrics with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_cer: Validation character error rate
            val_wer: Validation word error rate
            learning_rate: Current learning rate
            train_cer: Training CER (optional)
            train_wer: Training WER (optional)
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_cer'].append(val_cer)
        self.metrics['val_wer'].append(val_wer)
        self.metrics['learning_rates'].append(learning_rate)
        
        if train_cer is not None:
            self.metrics['train_cer'].append(train_cer)
        if train_wer is not None:
            self.metrics['train_wer'].append(train_wer)
    
    def save_metrics_json(self, filename: str = "training_metrics.json"):
        """Save metrics as JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_to_save[key] = value.tolist()
            elif isinstance(value, list):
                metrics_to_save[key] = value
            else:
                metrics_to_save[key] = value
        
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"✅ Saved metrics to {save_path}")
    
    def plot_loss_curves(self, save: bool = True, show: bool = False):
        """
        Plot training and validation loss curves.
        """
        if not self.metrics['epochs']:
            print("No metrics available to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Training loss
        plt.plot(self.metrics['epochs'], self.metrics['train_loss'], 
                'b-', label='Train Loss', linewidth=2)
        
        # Add validation loss if available (we don't have val_loss in current setup)
        # plt.plot(self.metrics['epochs'], self.metrics['val_loss'], 
        #         'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add vertical line for best model
        if self.metrics['val_cer']:
            best_idx = np.argmin(self.metrics['val_cer'])
            best_epoch = self.metrics['epochs'][best_idx]
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
                       label=f'Best Model (Epoch {best_epoch})')
            plt.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'loss_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved loss curves to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_cer_wer_curves(self, save: bool = True, show: bool = False):
        """
        Plot CER and WER curves for validation.
        """
        if not self.metrics['epochs']:
            print("No metrics available to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # CER plot
        axes[0].plot(self.metrics['epochs'], self.metrics['val_cer'], 
                    'r-', label='Val CER', linewidth=2)
        
        if self.metrics.get('train_cer'):
            axes[0].plot(self.metrics['epochs'], self.metrics['train_cer'], 
                        'b-', label='Train CER', linewidth=2, alpha=0.7)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('CER', fontsize=12)
        axes[0].set_title('Character Error Rate', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # WER plot
        axes[1].plot(self.metrics['epochs'], self.metrics['val_wer'], 
                    'r-', label='Val WER', linewidth=2)
        
        if self.metrics.get('train_wer'):
            axes[1].plot(self.metrics['epochs'], self.metrics['train_wer'], 
                        'b-', label='Train WER', linewidth=2, alpha=0.7)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('WER', fontsize=12)
        axes[1].set_title('Word Error Rate', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Find best epoch
        best_idx = np.argmin(self.metrics['val_cer'])
        best_epoch = self.metrics['epochs'][best_idx]
        best_cer = self.metrics['val_cer'][best_idx]
        
        # Add annotation
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        axes[0].annotate(f'Best CER: {best_cer:.4f}\n(Epoch {best_epoch})',
                        xy=(best_epoch, best_cer),
                        xytext=(best_epoch + 0.5, best_cer + 0.1),
                        fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'cer_wer_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved CER/WER curves to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_learning_rate(self, save: bool = True, show: bool = False):
        """
        Plot learning rate schedule.
        """
        if not self.metrics['epochs']:
            print("No metrics available to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.metrics['epochs'], self.metrics['learning_rates'], 
                    'g-', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'learning_rate.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved learning rate plot to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_all(self, save: bool = True, show: bool = False):
        """
        Generate all plots.
        """
        self.plot_loss_curves(save=save, show=show)
        self.plot_cer_wer_curves(save=save, show=show)
        self.plot_learning_rate(save=save, show=show)
        
        # Save metrics as JSON
        self.save_metrics_json()
    
    def plot_confusion_matrix(self, conf_matrix, class_names, save: bool = True):
        """
        Plot confusion matrix for character predictions.
        
        Args:
            conf_matrix: Confusion matrix (num_classes, num_classes)
            class_names: List of class names
        """
        # Only plot if matrix is not too large
        if conf_matrix.shape[0] > 30:
            print("Confusion matrix too large to plot (>30 classes)")
            return
        
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, cmap='Blues', aspect='auto')
        plt.colorbar()
        
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        
        # Add text annotations for small matrices
        if conf_matrix.shape[0] <= 20:
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, f'{conf_matrix[i, j]:.0f}',
                            ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved confusion matrix to {save_path}")
        
        plt.close()


def load_metrics_from_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Load metrics from a saved checkpoint file.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
    
    Returns:
        Dictionary containing training metrics
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    metrics = {
        'epoch': checkpoint.get('epoch', 0),
        'val_cer': checkpoint.get('val_cer', 0.0),
        'val_wer': checkpoint.get('val_wer', 0.0),
        'config': checkpoint.get('config', {})
    }
    
    return metrics


def generate_report_summary(metrics_history: Dict) -> str:
    """
    Generate a text summary of training results.
    
    Args:
        metrics_history: Dictionary with training metrics
    
    Returns:
        Formatted summary string
    """
    if not metrics_history['val_cer']:
        return "No metrics available."
    
    best_idx = np.argmin(metrics_history['val_cer'])
    best_epoch = metrics_history['epochs'][best_idx]
    best_cer = metrics_history['val_cer'][best_idx]
    best_wer = metrics_history['val_wer'][best_idx]
    
    summary = f"""
    ========================================
    TRAINING SUMMARY
    ========================================
    Total Epochs:        {len(metrics_history['epochs'])}
    Best Epoch:          {best_epoch}
    Best CER:            {best_cer:.4f}
    Best WER:            {best_wer:.4f}
    Final CER:           {metrics_history['val_cer'][-1]:.4f}
    Final WER:           {metrics_history['val_wer'][-1]:.4f}
    Final Learning Rate: {metrics_history['learning_rates'][-1]:.6f}
    ========================================
    """
    return summary


if __name__ == "__main__":
    # Test the visualizer with dummy data
    from pathlib import Path
    import numpy as np
    
    # Create dummy metrics
    visualizer = TrainingVisualizer(Path("./experiments/figures"))
    
    for epoch in range(1, 21):
        train_loss = 5.0 * np.exp(-epoch/10) + 0.5
        val_cer = 0.9 * np.exp(-epoch/15) + 0.1
        val_wer = 0.95 * np.exp(-epoch/15) + 0.15
        lr = 0.001 * (0.9 ** (epoch // 5))
        
        visualizer.update(epoch, train_loss, val_cer, val_wer, lr)
    
    # Generate plots
    visualizer.plot_all(save=True, show=False)
    
    # Generate summary
    summary = generate_report_summary(visualizer.metrics)
    print(summary)
    
    print("\n✅ Visualization test complete!")