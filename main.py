# gui_tf_trainer_tfmodels.py
# Full GUI that uses tf-models-official train_lib.run_experiment
# NOTE: adapt experiment_name to the task you want (default "image_classification_imagenet")

import os
import sys
import json
import yaml
import threading
import subprocess
import time
import copy
from typing import Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QPlainTextEdit, QLabel, QMessageBox, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, 
    QGridLayout, QProgressBar, QToolBar, QLineEdit,QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWebEngineWidgets import QWebEngineView
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
import numpy as np
import tensorflow_models as tfm

import cv2
import albumentations as A
ALBU_AVAILABLE = True


import tensorflow as tf
TF_AVAILABLE = True
from official.core import exp_factory, train_lib
from official.core import config_definitions as cfg_def  # for constructing base configs if needed

# ---------------------------
# Helper function to convert config dict to ParameterTree structure
# ---------------------------

# Custom widget for directory-only browsing (no file button)
class DirectoryOnlyBrowseWidget(QWidget):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sigChanged = None  # No change signal needed for this custom widget
        
        # Set minimum height to ensure widget stays visible
        self.setMinimumHeight(25)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Text field to show current path
        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(param.value()))
        self.lineEdit.textChanged.connect(self._on_text_changed)
        
        # Browse directory button only
        browse_dir_btn = QPushButton("Dir...")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setMinimumWidth(50)
        browse_dir_btn.clicked.connect(self._browse_directory)
        
        layout.addWidget(self.lineEdit)
        layout.addWidget(browse_dir_btn)
    
    def _on_text_changed(self, text):
        """Handle manual text changes in the line edit."""
        self.param.setValue(text)
    
    def _browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.param.name()}")
        if directory:
            self.lineEdit.setText(directory)
            self.param.setValue(directory)
    
    def value(self):
        """Return current value from the line edit."""
        return self.lineEdit.text()
    
    def setValue(self, value):
        """Set the value in the line edit."""
        self.lineEdit.setText(str(value))
    
    def focusInEvent(self, event):
        """Handle focus in event - ensure widget stays visible."""
        super().focusInEvent(event)
        self.lineEdit.setFocus()
    
    def focusOutEvent(self, event):
        """Handle focus out event - ensure widget stays visible."""
        super().focusOutEvent(event)
        # Don't hide the widget when losing focus
        self.show()

class DirectoryOnlyParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        widget = DirectoryOnlyBrowseWidget(self.param)
        # Ensure the widget is always visible
        widget.setVisible(True)
        return widget
    
    def valueChanged(self, param, data, info=None, force=False):
        """Handle external value changes."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.setValue(data)
            # Ensure widget stays visible after value change
            self.widget.show()
    
    def showEditor(self):
        """Override to ensure the widget is always shown."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
            return True
        return super().showEditor()
    
    def hideEditor(self):
        """Override to prevent hiding the widget."""
        # Don't actually hide the widget, just ensure it's visible
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
        return True

# Custom parameter type for directory-only browsing
class DirectoryOnlyParameter(pTypes.SimpleParameter):
    itemClass = DirectoryOnlyParameterItem

# Custom widget for directory/file browsing with buttons on same row
class DirectoryBrowseWidget(QWidget):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sigChanged = None  # No change signal needed for this custom widget
        
        # Set minimum height to ensure widget stays visible
        self.setMinimumHeight(25)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Text field to show current path
        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(param.value()))
        self.lineEdit.textChanged.connect(self._on_text_changed)
        
        # Browse directory button
        browse_dir_btn = QPushButton("Dir...")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setMinimumWidth(50)
        browse_dir_btn.clicked.connect(self._browse_directory)
        
        # Browse file button
        browse_file_btn = QPushButton("File...")
        browse_file_btn.setMaximumWidth(50)
        browse_file_btn.setMinimumWidth(50)
        browse_file_btn.clicked.connect(self._browse_file)
        
        layout.addWidget(self.lineEdit)
        layout.addWidget(browse_dir_btn)
        layout.addWidget(browse_file_btn)
    
    def _on_text_changed(self, text):
        """Handle manual text changes in the line edit."""
        self.param.setValue(text)
    
    def _browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.param.name()}")
        if directory:
            self.lineEdit.setText(directory)
            self.param.setValue(directory)
    
    def _browse_file(self):
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(None, f"Select file for {self.param.name()}")
        if file_path:
            self.lineEdit.setText(file_path)
            self.param.setValue(file_path)
    
    def value(self):
        """Return current value from the line edit."""
        return self.lineEdit.text()
    
    def setValue(self, value):
        """Set the value in the line edit."""
        self.lineEdit.setText(str(value))
    
    def focusInEvent(self, event):
        """Handle focus in event - ensure widget stays visible."""
        super().focusInEvent(event)
        self.lineEdit.setFocus()
    
    def focusOutEvent(self, event):
        """Handle focus out event - ensure widget stays visible."""
        super().focusOutEvent(event)
        # Don't hide the widget when losing focus
        self.show()

class DirectoryParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        widget = DirectoryBrowseWidget(self.param)
        # Ensure the widget is always visible
        widget.setVisible(True)
        return widget
    
    def valueChanged(self, param, data, info=None, force=False):
        """Handle external value changes."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.setValue(data)
            # Ensure widget stays visible after value change
            self.widget.show()
    
    def showEditor(self):
        """Override to ensure the widget is always shown."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
            return True
        return super().showEditor()
    
    def hideEditor(self):
        """Override to prevent hiding the widget."""
        # Don't actually hide the widget, just ensure it's visible
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
        return True

# Custom parameter type for directory browsing
class DirectoryParameter(pTypes.SimpleParameter):
    itemClass = DirectoryParameterItem

# Register the custom parameter types
pTypes.registerParameterType('directory', DirectoryParameter, override=True)
pTypes.registerParameterType('directory_only', DirectoryOnlyParameter, override=True)

def dict_to_params(data, name="Config"):
    """Convert a nested dictionary to Parameter tree structure."""
    if isinstance(data, dict):
        children = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary - create a group
                children.append(dict_to_params(value, key))
            else:
                # Check for special directory parameters
                if key in ['train_dir', 'val_dir'] and isinstance(value, str):
                    # Create directory/file browser parameter with both buttons
                    children.append({
                        'name': key,
                        'type': 'directory',
                        'value': value
                    })
                elif key in ['model_dir'] and isinstance(value, str):
                    # Create directory-only browser parameter (no file button)
                    children.append({
                        'name': key,
                        'type': 'directory_only',
                        'value': value
                    })
                else:
                    # Leaf value - determine type and create parameter
                    param_type = 'str'  # default
                    if isinstance(value, bool):
                        param_type = 'bool'
                    elif isinstance(value, int):
                        param_type = 'int'
                    elif isinstance(value, float):
                        param_type = 'float'
                    
                    children.append({
                        'name': key,
                        'type': param_type,
                        'value': value
                    })
        
        return {
            'name': name,
            'type': 'group',
            'children': children
        }
    else:
        # Single value
        param_type = 'str'
        if isinstance(data, bool):
            param_type = 'bool'
        elif isinstance(data, int):
            param_type = 'int'
        elif isinstance(data, float):
            param_type = 'float'
        
        return {
            'name': name,
            'type': param_type,
            'value': data
        }

def params_to_dict(param):
    """Convert Parameter tree back to dictionary."""
    if param.hasChildren():
        result = {}
        for child in param.children():
            child_name = child.name()
            result[child_name] = params_to_dict(child)
        return result
    else:
        return param.value()

# ---------------------------
# Bridge: GUI <-> Callbacks
# ---------------------------
class Bridge(QObject):
    log = Signal(str)
    update_plots = Signal(int, list, list, list, list)  # epoch_count, t_loss, v_loss, t_acc, v_acc
    progress = Signal(int)
    finished = Signal()

BRIDGE = Bridge()

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

# ---------------------------
# Helper: map GUI config -> exp_config
# This is a defensive mapping because different experiments expect different fields.
# You should adapt this function to match the exact exp_name you will use.
# ---------------------------
def map_gui_to_expconfig(gui_cfg: Dict[str, Any], exp_name: str):
    """
    Returns a ConfigDict from exp_factory.get_exp_config(exp_name) with fields
    updated from gui_cfg where sensible.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow / tf-models-official not available")

    exp_cfg = exp_factory.get_exp_config(exp_name)  # ConfigDict

    # Basic mapping examples (for image classification experiments)
    # Set model_dir / runtime
    model_dir = gui_cfg.get("train", {}).get("model_dir") or gui_cfg.get("train", {}).get("model_dir", "./model_dir")
    try:
        exp_cfg.runtime.model_dir = model_dir
    except Exception:
        # If runtime or model_dir not present, create
        try:
            exp_cfg.runtime = exp_cfg.get("runtime", {})
            exp_cfg.runtime.model_dir = model_dir
        except Exception:
            pass

    # Data paths: many official configs expect TFRecord input_path; we set whatever available
    train_dir = gui_cfg.get("data", {}).get("train_dir", "")
    val_dir = gui_cfg.get("data", {}).get("val_dir", "")

    # We will put the GUI paths into config.task.train_data.input_path and validation_data.input_path
    try:
        exp_cfg.task.train_data.input_path = train_dir
    except Exception:
        try:
            exp_cfg.task.train_data = exp_cfg.task.get("train_data", {})
            exp_cfg.task.train_data.input_path = train_dir
        except Exception:
            pass
    try:
        exp_cfg.task.validation_data.input_path = val_dir
    except Exception:
        try:
            exp_cfg.task.validation_data = exp_cfg.task.get("validation_data", {})
            exp_cfg.task.validation_data.input_path = val_dir
        except Exception:
            pass

    # batch size / image size mapping
    bs = gui_cfg.get("input", {}).get("batch_size")
    if bs is not None:
        try:
            exp_cfg.task.train_data.global_batch_size = int(bs)
        except Exception:
            try:
                exp_cfg.task.train_data.batch_size = int(bs)
            except Exception:
                pass

    image_size = gui_cfg.get("input", {}).get("image_size")
    if image_size is not None:
        try:
            exp_cfg.task.model.input_size = int(image_size)
        except Exception:
            # fallback: many configs have task.input_size / model.input_size etc.
            try:
                exp_cfg.task.input_size = int(image_size)
            except Exception:
                pass

    # training params
    epochs = gui_cfg.get("train", {}).get("epochs")
    if epochs is not None:
        try:
            exp_cfg.trainer.train_steps = int(epochs)  # may not be semantically correct for all experiments
        except Exception:
            # some configs use trainer.epochs
            try:
                exp_cfg.trainer.epochs = int(epochs)
            except Exception:
                pass

    lr = gui_cfg.get("train", {}).get("learning_rate")
    if lr is not None:
        try:
            # Many configs use trainer.optimizer_config.learning_rate.* structure; we set a simple value
            exp_cfg.trainer.optimizer_config.learning_rate.constant = float(lr)
        except Exception:
            try:
                exp_cfg.task.optimizer.learning_rate = float(lr)
            except Exception:
                pass

    # augmentation: place into task.train_data.augmentation (user pipeline will need to read and apply)
    aug = gui_cfg.get("augment", {})
    try:
        exp_cfg.task.train_data.augmentation = aug
    except Exception:
        try:
            exp_cfg.task.train_data = exp_cfg.task.get("train_data", {})
            exp_cfg.task.train_data.augmentation = aug
        except Exception:
            pass

    # callbacks: ensure QtBridgeCallback is present (we'll add it later before training)
    # set checkpoint resume path placeholder (init_checkpoint)
    return exp_cfg

# ---------------------------
# Trainer thread that calls train_lib.run_experiment
# ---------------------------
class TFModelsTrainerThread(threading.Thread):
    def __init__(self, gui_cfg: Dict[str, Any], exp_name: str = "image_classification_imagenet", resume_ckpt: str = None):
        super().__init__()
        self.gui_cfg = copy.deepcopy(gui_cfg)
        self.exp_name = exp_name
        self.resume_ckpt = resume_ckpt
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):

        try:
            BRIDGE.log.emit(f"Building exp_config for '{self.exp_name}' ...")
            exp_cfg = map_gui_to_expconfig(self.gui_cfg, self.exp_name)

            # set init_checkpoint if resume path provided
            if self.resume_ckpt:
                try:
                    exp_cfg.task.init_checkpoint = self.resume_ckpt
                    BRIDGE.log.emit(f"Set init_checkpoint: {self.resume_ckpt}")
                except Exception:
                    pass

            # ensure model_dir
            model_dir = self.gui_cfg.get("train", {}).get("model_dir", "./model_dir")
            try:
                exp_cfg.runtime.model_dir = model_dir
            except Exception:
                try:
                    exp_cfg.runtime = exp_cfg.get("runtime", {})
                    exp_cfg.runtime.model_dir = model_dir
                except Exception:
                    pass
            os.makedirs(model_dir, exist_ok=True)

            # Add our QtBridgeCallback into exp_cfg.callbacks (avoid duplicates)
            total_steps = int(self.gui_cfg.get("train", {}).get("epochs", 1))
            cb = {"type": "QtBridgeCallback", "total_train_steps": total_steps, "log_every_n": 1}
            # ensure callbacks list exists
            try:
                if hasattr(exp_cfg, "callbacks") and exp_cfg.callbacks:
                    # remove previous QtBridgeCallback entries
                    exp_cfg.callbacks = [c for c in exp_cfg.callbacks if not (isinstance(c, dict) and c.get("type") == "QtBridgeCallback")]
                    exp_cfg.callbacks.append(cb)
                else:
                    exp_cfg.callbacks = [cb]
            except Exception:
                try:
                    exp_cfg.callbacks = [cb]
                except Exception:
                    pass

            BRIDGE.log.emit("Starting train_lib.run_experiment ...")
            # Note: some run_experiment wrappers accept distribution_strategy as arg; we pass runtime setting if present
            ds = None
            distribution = None
            try:
                distribution = getattr(exp_cfg.runtime, "distribution_strategy", None)
            except Exception:
                distribution = None

            # Run training. This is a blocking call.
            train_lib.run_experiment(
                distribution_strategy=distribution or "mirrored",
                mode="train",
                params=exp_cfg
            )

            BRIDGE.log.emit("train_lib.run_experiment returned (training finished).")
        except Exception as e:
            BRIDGE.log.emit(f"Training exception: {e}")
        finally:
            BRIDGE.finished.emit()

# ---------------------------
# MainWindow: similar UI as earlier, but start TFModelsTrainerThread
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self, experiment_name: str = "image_classification_imagenet"):
        super().__init__()
        self.setWindowTitle("TF-Models-Official GUI Trainer")
        self.resize(1600, 1000)

        # initialize GUI config (this is the smaller GUI-level config)
        self.gui_cfg = {
            "data": {"train_dir": "", "val_dir": ""},
            "input": {"image_size": 224, "batch_size": 16, "shuffle": True},
            "train": {"epochs": 5, "learning_rate": 1e-3, "model_dir": "./model_dir"},
            "augment": {"flip_horizontal": True, "flip_vertical": False, "rotate_limit": 15, "random_crop_pct": 0.1,
                        "brightness_limit": 0.2, "contrast_limit": 0.2}
        }
        self.experiment_name = experiment_name
        self.trainer_thread: TFModelsTrainerThread = None
        self.resume_ckpt_path = None
        self.tb_proc = None

        # toolbar
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        act_save_json = QAction("Save JSON", self); act_save_json.triggered.connect(lambda: self.save_config("json"))
        toolbar.addAction(act_save_json)
        act_save_yaml = QAction("Save YAML", self); act_save_yaml.triggered.connect(lambda: self.save_config("yaml"))
        toolbar.addAction(act_save_yaml)
        act_load = QAction("Load Config", self); act_load.triggered.connect(self.load_config)
        toolbar.addAction(act_load)
        toolbar.addSeparator()
        act_ckpt = QAction("Choose checkpoint", self); act_ckpt.triggered.connect(self.choose_checkpoint)
        toolbar.addAction(act_ckpt)
        act_model_dir = QAction("Choose model_dir", self); act_model_dir.triggered.connect(self.choose_model_dir)
        toolbar.addAction(act_model_dir)

        # left layout: config tree + augment controls + controls
        left_layout = QVBoxLayout()
        
        # Create ParameterTree with config data
        self.params = Parameter.create(**dict_to_params(self.gui_cfg, "Configuration"))
        self.tree = ParameterTree()
        self.tree.setParameters(self.params, showTop=False)
        
        # Set up directory parameter callbacks
        self._setup_directory_callbacks()
        
        # Connect to parameter change signals
        self.params.sigTreeStateChanged.connect(self._on_param_changed)
        
        left_layout.addWidget(QLabel("Config")); left_layout.addWidget(self.tree, stretch=3)

        # augmentation panel
        aug_form = QFormLayout()
        self.chk_flip_h = QCheckBox(); self.chk_flip_h.setChecked(self.gui_cfg["augment"]["flip_horizontal"])
        aug_form.addRow("Flip H", self.chk_flip_h)
        self.chk_flip_v = QCheckBox(); self.chk_flip_v.setChecked(self.gui_cfg["augment"]["flip_vertical"])
        aug_form.addRow("Flip V", self.chk_flip_v)
        self.spin_rotate = QSpinBox(); self.spin_rotate.setRange(0,90); self.spin_rotate.setValue(self.gui_cfg["augment"]["rotate_limit"])
        aug_form.addRow("Rotate limit", self.spin_rotate)
        self.spin_crop = QDoubleSpinBox(); self.spin_crop.setRange(0.0,0.5); self.spin_crop.setSingleStep(0.01)
        self.spin_crop.setValue(self.gui_cfg["augment"]["random_crop_pct"]); aug_form.addRow("Random crop pct", self.spin_crop)
        self.spin_bright = QDoubleSpinBox(); self.spin_bright.setRange(0.0,1.0); self.spin_bright.setSingleStep(0.01)
        self.spin_bright.setValue(self.gui_cfg["augment"]["brightness_limit"]); aug_form.addRow("Brightness limit", self.spin_bright)
        self.spin_contrast = QDoubleSpinBox(); self.spin_contrast.setRange(0.0,1.0); self.spin_contrast.setSingleStep(0.01)
        self.spin_contrast.setValue(self.gui_cfg["augment"]["contrast_limit"]); aug_form.addRow("Contrast limit", self.spin_contrast)

        btn_preview = QPushButton("Preview Aug"); btn_preview.clicked.connect(self.preview_augmentation)
        aug_form.addRow(btn_preview)

        left_layout.addLayout(aug_form)

        # training controls (directory selection is now integrated in parameter tree)
        self.btn_start = QPushButton("Start Training"); self.btn_start.clicked.connect(self.start_training); left_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop Training"); self.btn_stop.clicked.connect(self.stop_training); left_layout.addWidget(self.btn_stop)

        self.progress = QProgressBar(); left_layout.addWidget(self.progress)

        left_widget = QWidget(); left_widget.setLayout(left_layout)

        # right layout: TensorBoard + logs + plots
        right_layout = QVBoxLayout()
        self.tb_view = QWebEngineView()
        
        self.tb_view.setUrl(f"http://localhost:6006")  # Default URL for TensorBoard
        right_layout.addWidget(QLabel("TensorBoard"))
        right_layout.addWidget(self.tb_view, stretch=2)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(QLabel("Logs"))
        right_layout.addWidget(self.log_edit, stretch=1)
        self.plot = pg.PlotWidget(title="Metrics")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True)
        self.curves = {"train_loss": self.plot.plot(pen='r', name="train_loss"), "val_loss": self.plot.plot(pen='b', name="val_loss"),
                       "train_acc": self.plot.plot(pen='g', name="train_acc"), "val_acc": self.plot.plot(pen='y', name="val_acc")}
        # right_layout.addWidget(self.plot, stretch=2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(right_widget, stretch=2)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # signals
        BRIDGE.log.connect(self.append_log); BRIDGE.update_plots.connect(self.on_update_plots); BRIDGE.progress.connect(self.progress.setValue)
        BRIDGE.finished.connect(self.on_training_finished)

        # state
        self.resume_ckpt_path = None

    def closeEvent(self, event):
        """Handle window closing to clean up resources."""
        super().closeEvent(event)

    # tree helpers
    def _setup_directory_callbacks(self):
        """Set up callbacks for directory browser parameters."""
        # With the new DirectoryParameter implementation, changes are handled
        # automatically through the sigValueChanged signal, so we don't need
        # special callback setup here anymore
        pass
    
    def _on_param_changed(self, param, changes):
        """Handle parameter changes from the ParameterTree."""
        try:
            # Update gui_cfg when parameters change
            self.gui_cfg = params_to_dict(self.params)
            self.append_log("Config updated from parameter tree")
            # Update other UI elements if needed
            self.apply_cfg_to_widgets()
        except Exception as e:
            self.append_log(f"Error updating config from parameters: {e}")
    
    def refresh_tree(self):
        """Update the ParameterTree with current config data."""
        try:
            # Recreate the parameter structure
            new_params = Parameter.create(**dict_to_params(self.gui_cfg, "Configuration"))
            
            # Disconnect old signals
            if hasattr(self, 'params'):
                self.params.sigTreeStateChanged.disconnect()
            
            # Set new parameters
            self.params = new_params
            self.tree.setParameters(self.params, showTop=False)
            
            # Set up directory callbacks
            self._setup_directory_callbacks()
            
            # Reconnect signals
            self.params.sigTreeStateChanged.connect(self._on_param_changed)
            
        except Exception as e:
            self.append_log(f"Error refreshing tree: {e}")
    
    def write_back_tree(self):
        """Extract data from the parameter tree back to gui_cfg."""
        try:
            self.gui_cfg = params_to_dict(self.params)
            self.append_log("Config updated from parameter tree")
        except Exception as e:
            self.append_log(f"Could not write back tree data: {e}")

    # file ops
    def save_config(self, fmt="json"):
        self.sync_aug_to_cfg()
        path, _ = QFileDialog.getSaveFileName(self, "Save config", filter=f"*.{fmt}")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(self.gui_cfg, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(self.gui_cfg, f, allow_unicode=True)
        QMessageBox.information(self, "Saved", f"Saved config to {path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", filter="*.json *.yaml *.yml")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                self.gui_cfg = json.load(f)
            else:
                self.gui_cfg = yaml.safe_load(f)
        self.apply_cfg_to_widgets()
        self.refresh_tree()
        QMessageBox.information(self, "Loaded", f"Loaded: {path}")

    # data/model path
    def set_train_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select train_dir")
        if d:
            self.gui_cfg["data"]["train_dir"] = d
            self.refresh_tree()

    def set_val_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select val_dir")
        if d:
            self.gui_cfg["data"]["val_dir"] = d
            self.refresh_tree()

    def choose_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint (ckpt prefix or file)")
        if p:
            self.resume_ckpt_path = p
            self.append_log(f"Will resume from: {p}")

    def choose_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose model_dir")
        if d:
            self.gui_cfg["train"]["model_dir"] = d
            self.refresh_tree()

    # augment panel sync
    def apply_cfg_to_widgets(self):
        aug = self.gui_cfg.get("augment", {})
        self.chk_flip_h.setChecked(bool(aug.get("flip_horizontal", False)))
        self.chk_flip_v.setChecked(bool(aug.get("flip_vertical", False)))
        self.spin_rotate.setValue(int(aug.get("rotate_limit", 0)))
        self.spin_crop.setValue(float(aug.get("random_crop_pct", 0.0)))
        self.spin_bright.setValue(float(aug.get("brightness_limit", 0.0)))
        self.spin_contrast.setValue(float(aug.get("contrast_limit", 0.0)))

    def sync_aug_to_cfg(self):
        self.gui_cfg.setdefault("augment", {})
        self.gui_cfg["augment"].update({
            "flip_horizontal": bool(self.chk_flip_h.isChecked()),
            "flip_vertical": bool(self.chk_flip_v.isChecked()),
            "rotate_limit": int(self.spin_rotate.value()),
            "random_crop_pct": float(self.spin_crop.value()),
            "brightness_limit": float(self.spin_bright.value()),
            "contrast_limit": float(self.spin_contrast.value()),
        })
        self.refresh_tree()
    
    def update_config_value(self, path_str, value):
        """Update a config value by dot-separated path string."""
        try:
            path = path_str.split('.')
            current = self.gui_cfg
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = value
            self.refresh_tree()
            self.append_log(f"Updated config: {path_str} = {value}")
        except Exception as e:
            self.append_log(f"Error updating config {path_str}: {e}")

    # preview augmentation (uses albumentations)
    def preview_augmentation(self):
        if not ALBU_AVAILABLE:
            QMessageBox.warning(self, "Missing libs", "Install albumentations/opencv-python for preview.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose image for preview")
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "Read failed", "Cannot read image.")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.sync_aug_to_cfg()
        size = int(self.gui_cfg["input"].get("image_size", 224))
        pipe = build_albu_pipeline(self.gui_cfg["augment"], size, size)
        samples = []
        for _ in range(4):
            try:
                aug = pipe(image=img_rgb)["image"]
                if aug.dtype != np.uint8 and np.max(aug) <= 1.1:
                    aug = np.clip(aug * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                aug = cv2.resize(img_rgb, (size, size))
            samples.append(aug)
        dlg = QDialog(self)
        dlg.setWindowTitle("Aug Preview")
        layout = QGridLayout(dlg)
        pix = np_to_qpixmap(img_rgb)
        w0 = QLabel(); w0.setPixmap(pix.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel("Original"),0,0); layout.addWidget(w0,1,0)
        for i, s in enumerate(samples):
            p = np_to_qpixmap(s); l = QLabel(); l.setPixmap(p.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel(f"Aug {i+1}"),0,i+1); layout.addWidget(l,1,i+1)
        dlg.exec()

    # logging / plotting handlers
    def append_log(self, text):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {text}")

    def on_update_plots(self, epoch, tl, vl, ta, va):
        x = list(range(1, epoch+1))
        try:
            self.curves["train_loss"].setData(x, tl)
            self.curves["val_loss"].setData(x, vl)
            self.curves["train_acc"].setData(x, ta)
            self.curves["val_acc"].setData(x, va)
        except Exception:
            pass

    # tensorboard
    def start_tensorboard(self, model_dir):
        try:
            port = 6006
            if self.tb_proc and getattr(self.tb_proc, "poll", None) is None and self.tb_proc.poll() is None:
                pass
            else:
                self.tb_proc = subprocess.Popen(["tensorboard", "--logdir", model_dir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.tb_view.setUrl(f"http://localhost:{port}")
            self.append_log(f"TensorBoard -> {model_dir}")
        except Exception as e:
            self.append_log(f"Failed to start TensorBoard: {e}")

    # training controls
    def start_training(self):
        if not TF_AVAILABLE:
            QMessageBox.critical(self, "TF missing", "TensorFlow or tf-models-official not available.")
            return
        # sync GUI config
        self.sync_aug_to_cfg()
        self.write_back_tree()
        model_dir = self.gui_cfg["train"].get("model_dir", "./model_dir")
        os.makedirs(model_dir, exist_ok=True)
        self.start_tensorboard(model_dir)
        # start trainer thread that runs train_lib.run_experiment
        self.append_log("Launching tf-models-official trainer thread...")
        t = TFModelsTrainerThread(self.gui_cfg, exp_name=self.experiment_name, resume_ckpt=self.resume_ckpt_path)
        self.trainer_thread = t
        t.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

    def stop_training(self):
        if self.trainer_thread:
            self.append_log("Requested stop — trainer thread will attempt graceful stop.")
            self.trainer_thread.stop()
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def on_training_finished(self):
        self.append_log("Training finished (thread signalled).")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.progress.setValue(100)

# helpers used above (np_to_qpixmap, build_albu_pipeline) - include simple definitions here:
def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h,w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h,w,c = img.shape
        if c == 3:
            qimg = QImage(img.data, w, h, w*3, QImage.Format_RGB888)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if 'cv2' in globals() else img[:,:,0]
            h,w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())

def build_albu_pipeline(aug_cfg: Dict[str,Any], target_h:int, target_w:int):
    if not ALBU_AVAILABLE:
        raise RuntimeError("albumentations / cv2 missing")
    transforms = []
    if aug_cfg.get("flip_horizontal"):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("flip_vertical"):
        transforms.append(A.VerticalFlip(p=0.5))
    rot = aug_cfg.get("rotate_limit", 0)
    if rot and rot > 0:
        transforms.append(A.Rotate(limit=rot, border_mode=cv2.BORDER_REFLECT_101, p=0.6))
    crop = aug_cfg.get("random_crop_pct", 0.0)
    if crop and 0.0 < crop <= 0.5:
        transforms.append(A.RandomResizedCrop(height=target_h, width=target_w, scale=(1.0-crop, 1.0), ratio=(0.9,1.1), p=0.6))
    bl = aug_cfg.get("brightness_limit", 0.0); cl = aug_cfg.get("contrast_limit", 0.0)
    if (bl and bl>0) or (cl and cl>0):
        transforms.append(A.RandomBrightnessContrast(brightness_limit=bl, contrast_limit=cl, p=0.6))
    transforms.append(A.Resize(target_h, target_w))
    transforms.append(A.Normalize())
    return A.Compose(transforms)

# run
def main():
    app = QApplication(sys.argv)
    win = MainWindow(experiment_name="image_classification_imagenet")
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
