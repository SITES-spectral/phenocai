"""Training callbacks for PhenoCAI."""

import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


def create_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    monitor: str = 'val_loss',
    mode: str = 'min',
    patience: int = 10,
    reduce_lr_patience: int = 5,
    save_best_only: bool = True,
    save_weights_only: bool = False,
    verbose: int = 1
) -> List[tf.keras.callbacks.Callback]:
    """Create standard callbacks for training.
    
    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        monitor: Metric to monitor
        mode: 'min' or 'max'
        patience: Early stopping patience
        reduce_lr_patience: ReduceLROnPlateau patience
        save_best_only: Save only best model
        save_weights_only: Save only weights
        verbose: Verbosity level
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Model checkpoint
    checkpoint_path = str(Path(checkpoint_dir) / 'best_model.weights.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=True,  # Force weights only to avoid pickle errors
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=verbose
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0
    )
    callbacks.append(tensorboard)
    
    return callbacks


class TrainingMonitor(tf.keras.callbacks.Callback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, log_dir: str, update_frequency: int = 10):
        """Initialize training monitor.
        
        Args:
            log_dir: Directory for logs
            update_frequency: How often to log (in batches)
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.update_frequency = update_frequency
        self.batch_times = []
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        print(f"\nTraining started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""
        self.epoch_start_time = datetime.now()
        self.batch_times = []
        
    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch."""
        self.batch_times.append(datetime.now())
        
        if batch % self.update_frequency == 0 and batch > 0:
            # Calculate statistics
            avg_batch_time = (
                sum((t2 - t1).total_seconds() for t1, t2 in 
                    zip(self.batch_times[:-1], self.batch_times[1:]))
                / len(self.batch_times[1:])
            )
            
            # Estimate time remaining
            total_batches = self.params.get('steps', 0)
            if total_batches > 0:
                remaining_batches = total_batches - batch
                eta_seconds = remaining_batches * avg_batch_time
                eta_minutes = int(eta_seconds // 60)
                eta_seconds = int(eta_seconds % 60)
                
                print(f"\rBatch {batch}/{total_batches} - "
                      f"Avg time: {avg_batch_time:.2f}s - "
                      f"ETA: {eta_minutes}m {eta_seconds}s", end='')
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        
        # Log metrics
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s - {metrics_str}")
        
        # Save metrics to file
        metrics_file = self.log_dir / 'training_metrics.csv'
        
        # Create header if file doesn't exist
        if not metrics_file.exists():
            headers = ['epoch', 'time'] + list(logs.keys())
            with open(metrics_file, 'w') as f:
                f.write(','.join(headers) + '\n')
        
        # Append metrics
        values = [str(epoch + 1), f"{epoch_time:.1f}"] + [f"{v:.6f}" for v in logs.values()]
        with open(metrics_file, 'a') as f:
            f.write(','.join(values) + '\n')
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
        
        # Save final summary
        summary_file = self.log_dir / 'training_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"================\n")
            f.write(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Total epochs: {self.params.get('epochs', 0)}\n")
            f.write(f"\nFinal metrics:\n")
            for k, v in logs.items():
                f.write(f"  {k}: {v:.6f}\n")


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """Callback to save confusion matrix during training."""
    
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        class_names: List[str],
        log_dir: str,
        frequency: int = 5
    ):
        """Initialize confusion matrix callback.
        
        Args:
            validation_data: Validation dataset
            class_names: List of class names
            log_dir: Directory for saving plots
            frequency: How often to generate (in epochs)
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        
    def on_epoch_end(self, epoch, logs=None):
        """Generate confusion matrix at end of epoch."""
        if (epoch + 1) % self.frequency != 0:
            return
            
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            if predictions.shape[1] == 1:  # Binary classification
                y_pred.extend((predictions > 0.5).astype(int).flatten())
            else:  # Multi-class
                y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = self.log_dir / f'confusion_matrix_epoch_{epoch + 1}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nConfusion matrix saved to {plot_path}")


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Log learning rate during training."""
    
    def __init__(self, log_dir: str):
        """Initialize learning rate logger.
        
        Args:
            log_dir: Directory for logs
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lr_file = self.log_dir / 'learning_rates.csv'
        
    def on_epoch_begin(self, epoch, logs=None):
        """Log learning rate at beginning of epoch."""
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        else:
            lr = tf.keras.backend.get_value(lr)
            
        # Create header if file doesn't exist
        if not self.lr_file.exists():
            with open(self.lr_file, 'w') as f:
                f.write('epoch,learning_rate\n')
        
        # Append learning rate
        with open(self.lr_file, 'a') as f:
            f.write(f'{epoch + 1},{lr:.8f}\n')