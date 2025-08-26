from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QSizePolicy
from PySide6.QtCore import QObject, Signal
import tensorflow as tf
import keras

class Bridge(QObject):
    log = Signal(str)
    update_plots = Signal(int, list, list, list, list)  # epoch_count, t_loss, v_loss, t_acc, v_acc
    progress = Signal(int)
    progress_bar = Signal(str, int, int, dict)  # step_info, current_step, total_steps, metrics
    finished = Signal()


# ---------------------------
# Qt callback class for tf-models-official
# ---------------------------

class QtBridgeCallback(keras.callbacks.Callback):
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
            except Exception:
                self._steps_per_epoch = 100
                
        # Log epoch start with better information
        total_epochs = getattr(self.params, 'epochs', '?') if hasattr(self, 'params') else '?'
        BRIDGE.log.emit(f"Starting Epoch {epoch + 1}/{total_epochs} ({self._steps_per_epoch} steps)")
            
    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        self._current_epoch_steps += 1
        
        # Overall progress for progress bar
        if self.total_train_steps:
            pct = int(min(100, (self._batch_count / max(1, self.total_train_steps)) * 100))
            BRIDGE.progress.emit(pct)
            
        # Emit detailed progress bar info for logs widget
        metrics = logs or {}
        step_info = f"Epoch {self._epoch + 1}"
        BRIDGE.progress_bar.emit(step_info, self._current_epoch_steps, self._steps_per_epoch, metrics)
        
        # Log progression every 10% of total batches in current epoch
        if self._steps_per_epoch > 0:
            progress_interval = max(1, self._steps_per_epoch // 10)  # 10% intervals
            if self._current_epoch_steps % progress_interval == 0:
                progress_pct = int((self._current_epoch_steps / self._steps_per_epoch) * 100)
                metrics_str = ""
                if logs:
                    loss = logs.get('loss', 0)
                    acc = logs.get('accuracy', logs.get('acc', 0))
                    if loss and acc:
                        metrics_str = f" - loss: {loss:.4f} - accuracy: {acc:.4f}"
                    elif loss:
                        metrics_str = f" - loss: {loss:.4f}"
                BRIDGE.log.emit(f"Epoch {self._epoch + 1} - {progress_pct}% ({self._current_epoch_steps}/{self._steps_per_epoch}){metrics_str}")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._epoch += 1
        tl = logs.get("loss", float("nan"))
        ta = logs.get("accuracy", logs.get("acc", float("nan")))
        vl = logs.get("val_loss", float("nan"))
        va = logs.get("val_accuracy", logs.get("val_acc", float("nan")))
        
        # Store for plotting
        self._train_losses.append(float(tl) if tl is not None else float("nan"))
        self._val_losses.append(float(vl) if vl is not None else float("nan"))
        self._train_accs.append(float(ta) if ta is not None else float("nan"))
        self._val_accs.append(float(va) if va is not None else float("nan"))
        
        # Send final progress bar for this epoch
        BRIDGE.progress_bar.emit(f"Epoch {self._epoch}", self._steps_per_epoch, self._steps_per_epoch, logs)
        
        # Log epoch summary  
        metrics_summary = []
        if not (tl != tl):  # Check if not NaN
            metrics_summary.append(f"loss: {tl:.4f}")
        if not (ta != ta):  # Check if not NaN
            metrics_summary.append(f"accuracy: {ta:.4f}")
        if not (vl != vl):  # Check if not NaN
            metrics_summary.append(f"val_loss: {vl:.4f}")
        if not (va != va):  # Check if not NaN
            metrics_summary.append(f"val_accuracy: {va:.4f}")
            
        summary_text = " - ".join(metrics_summary) if metrics_summary else "No metrics available"
        BRIDGE.log.emit(f"Epoch {self._epoch} completed - {summary_text}")
        
        # Update plots
        BRIDGE.update_plots.emit(self._epoch, self._train_losses, self._val_losses, self._train_accs, self._val_accs)
    def on_train_end(self, logs=None):
        BRIDGE.log.emit("Callback: training ended.")
        BRIDGE.finished.emit()


BRIDGE = Bridge()
