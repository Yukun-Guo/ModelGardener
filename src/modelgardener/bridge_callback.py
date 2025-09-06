import time
import keras


class Bridge:
    """Simple callback bridge for CLI mode - logs to console instead of GUI."""
    
    def log(self, message: str):
        """Log message to console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def update_plots(self, epoch_count, t_loss, v_loss, t_acc, v_acc):
        """Update plots - in CLI mode, just log metrics."""
        if epoch_count > 0:
            print(f"Epoch {epoch_count}: Train Loss: {t_loss[-1] if t_loss else 'N/A'}, "
                  f"Val Loss: {v_loss[-1] if v_loss else 'N/A'}, "
                  f"Train Acc: {t_acc[-1] if t_acc else 'N/A'}, "
                  f"Val Acc: {v_acc[-1] if v_acc else 'N/A'}")
    
    def progress(self, value: int):
        """Update progress - in CLI mode, just log progress."""
        print(f"Progress: {value}%")
    
    def progress_bar(self, step_info: str, current_step: int, total_steps: int, metrics: dict):
        """Update progress bar - in CLI mode, show progress."""
        percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0
        print(f"{step_info} - Step {current_step}/{total_steps} ({percentage:.1f}%) - {metrics}")
    
    def finished(self):
        """Signal training finished."""
        print("Training finished!")


# ---------------------------
# CLI callback class for tf-models-official
# ---------------------------

class CLIBridgeCallback(keras.callbacks.Callback):
    def __init__(self, total_train_steps: int = 1000, log_every_n: int = 1):
        super().__init__()
        self.total_train_steps = int(total_train_steps)
        self.log_every_n = int(log_every_n)
        self._epoch = 0
        self._current_epoch_steps = 0
        self._steps_per_epoch = 0
        # lists for plotting
        self._train_losses = []
        self._val_losses = []
        self._train_accs = []
        self._val_accs = []
        self._batch_count = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch_steps = 0
        # Better estimation of steps per epoch
        if hasattr(self, 'params') and self.params:
            self._steps_per_epoch = self.params.get('steps', 100)
        elif hasattr(self.model, 'steps_per_epoch'):
            self._steps_per_epoch = self.model.steps_per_epoch
        else:
            # Try to get from training data
            try:
                # Estimate from total_train_steps and epochs if available
                if hasattr(self.params, 'epochs') and self.params['epochs'] > 0:
                    self._steps_per_epoch = self.total_train_steps // self.params['epochs']
                elif hasattr(self.model, 'train_step_counter'):
                    self._steps_per_epoch = int(self.model.train_step_counter)
                else:
                    self._steps_per_epoch = 100  # fallback
            except AttributeError:
                self._steps_per_epoch = 100
        
        self._epoch = epoch
        total_epochs = getattr(self.params, 'epochs', 1) if hasattr(self, 'params') else 1
        BRIDGE.log(f"Starting Epoch {epoch + 1}/{total_epochs} ({self._steps_per_epoch} steps)")
    
    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        self._current_epoch_steps += 1
        
        # Update progress
        if self._steps_per_epoch > 0:
            pct = int((self._current_epoch_steps / self._steps_per_epoch) * 100)
            BRIDGE.progress(pct)
        
        # Create step info and metrics for progress bar
        step_info = f"Epoch {self._epoch + 1}"
        metrics = logs or {}
        BRIDGE.progress_bar(step_info, self._current_epoch_steps, self._steps_per_epoch, metrics)

        # Log every N batches within an epoch
        if self._current_epoch_steps % self.log_every_n == 0:
            progress_pct = int((self._current_epoch_steps / self._steps_per_epoch) * 100) if self._steps_per_epoch > 0 else 0
            
            metrics_str = ""
            if logs:
                metrics_parts = []
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        metrics_parts.append(f"{key}: {value:.4f}")
                    else:
                        metrics_parts.append(f"{key}: {value}")
                if metrics_parts:
                    metrics_str = f" - {', '.join(metrics_parts)}"
            
            BRIDGE.log(f"Epoch {self._epoch + 1} - {progress_pct}% ({self._current_epoch_steps}/{self._steps_per_epoch}){metrics_str}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Store metrics for plotting
        if 'loss' in logs:
            self._train_losses.append(logs['loss'])
        if 'val_loss' in logs:
            self._val_losses.append(logs['val_loss'])
        if 'accuracy' in logs:
            self._train_accs.append(logs['accuracy'])
        elif 'acc' in logs:
            self._train_accs.append(logs['acc'])
        if 'val_accuracy' in logs:
            self._val_accs.append(logs['val_accuracy'])
        elif 'val_acc' in logs:
            self._val_accs.append(logs['val_acc'])
        
        # Final progress bar for completed epoch
        BRIDGE.progress_bar(f"Epoch {self._epoch}", self._steps_per_epoch, self._steps_per_epoch, logs)
        
        # Create summary text
        summary_parts = []
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                summary_parts.append(f"{key}: {value:.4f}")
            else:
                summary_parts.append(f"{key}: {value}")
        
        summary_text = ", ".join(summary_parts) if summary_parts else "No metrics"
        BRIDGE.log(f"Epoch {self._epoch} completed - {summary_text}")
        
        # Send plot update
        BRIDGE.update_plots(self._epoch, self._train_losses, self._val_losses, self._train_accs, self._val_accs)
    
    def on_train_end(self, logs=None):
        BRIDGE.log("Callback: training ended.")
        BRIDGE.finished()


# Create global instance 
BRIDGE = Bridge()
