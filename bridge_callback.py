from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QSizePolicy
from PySide6.QtCore import QObject, Signal
import tensorflow as tf

class Bridge(QObject):
    log = Signal(str)
    update_plots = Signal(int, list, list, list, list)  # epoch_count, t_loss, v_loss, t_acc, v_acc
    progress = Signal(int)
    finished = Signal()


# ---------------------------
# Qt callback class for tf-models-official
# ---------------------------

class QtBridgeCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_train_steps: int = 1000, log_every_n: int = 1):
        super().__init__()
        self.total_train_steps = int(total_train_steps)
        self.log_every_n = int(log_every_n)
        self._epoch = 0
        # lists for plotting
        self._train_losses = []
        self._val_losses = []
        self._train_accs = []
        self._val_accs = []
        self._batch_count = 0
    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        if self.total_train_steps:
            pct = int(min(100, (self._batch_count / max(1, self.total_train_steps)) * 100))
            BRIDGE.progress.emit(pct)
        if self._batch_count % max(1, self.log_every_n) == 0:
            BRIDGE.log.emit(f"[batch {self._batch_count}] {logs or {}}")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._epoch += 1
        tl = logs.get("loss", float("nan"))
        ta = logs.get("accuracy", logs.get("acc", float("nan")))
        vl = logs.get("val_loss", float("nan"))
        va = logs.get("val_accuracy", logs.get("val_acc", float("nan")))
        self._train_losses.append(float(tl) if tl is not None else float("nan"))
        self._val_losses.append(float(vl) if vl is not None else float("nan"))
        self._train_accs.append(float(ta) if ta is not None else float("nan"))
        self._val_accs.append(float(va) if va is not None else float("nan"))
        BRIDGE.log.emit(f"[Epoch {self._epoch}] loss={tl:.4f} acc={ta:.4f} val_loss={vl} val_acc={va}")
        BRIDGE.update_plots.emit(self._epoch, self._train_losses, self._val_losses, self._train_accs, self._val_accs)
    def on_train_end(self, logs=None):
        BRIDGE.log.emit("Callback: training ended.")
        BRIDGE.finished.emit()


BRIDGE = Bridge()
