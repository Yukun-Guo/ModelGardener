import os
import json
import yaml
import subprocess
import time
import importlib.util
from typing import Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QPlainTextEdit, QLabel, QMessageBox, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, 
    QGridLayout, QProgressBar, QToolBar, QLineEdit, QSizePolicy,
    QTreeWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWebEngineWidgets import QWebEngineView
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from directory_only_parameter import DirectoryOnlyParameter
from directory_parameter import DirectoryParameter
from augmentation_group import AugmentationGroup
from preprocessing_group import PreprocessingGroup
from callbacks_group import CallbacksGroup
from loss_functions_group import LossFunctionsGroup
from metrics_group import MetricsGroup
from bridge_callback import BRIDGE
from trainer_thread import TFModelsTrainerThread
import pyqtgraph.parametertree.parameterTypes as pTypes
import numpy as np
import cv2
import albumentations as A

# Register the custom parameter types
pTypes.registerParameterType('directory', DirectoryParameter, override=True)
pTypes.registerParameterType('directory_only', DirectoryOnlyParameter, override=True)
pTypes.registerParameterType('augmentation_group', AugmentationGroup, override=True)
pTypes.registerParameterType('preprocessing_group', PreprocessingGroup, override=True)
pTypes.registerParameterType('callbacks_group', CallbacksGroup, override=True)
pTypes.registerParameterType('loss_functions_group', LossFunctionsGroup, override=True)
pTypes.registerParameterType('metrics_group', MetricsGroup, override=True)


# ---------------------------
# MainWindow: similar UI as earlier, but start TFModelsTrainerThread
# ---------------------------
class MainWindow(QMainWindow):
    
    def __init__(self, experiment_name: str = "image_classification_imagenet"):
        super().__init__()
        self.setWindowTitle("TF-Models-Official GUI Trainer")
        self.resize(1600, 1000)

        # initialize GUI config (comprehensive TensorFlow Models config)
        comprehensive_config = self.create_comprehensive_config()
        self.gui_cfg = {
            **comprehensive_config['basic'],
            **comprehensive_config['advanced']
        }
        
        # Also maintain the comprehensive structure for the parameter tree
        self.comprehensive_cfg = comprehensive_config
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
        
        # Create ParameterTree with comprehensive config data organized in Basic/Advanced sections
        self.params = Parameter.create(**self.dict_to_params(self.comprehensive_cfg, "Configuration"))
        self.tree = ParameterTree()
        self.tree.setParameters(self.params, showTop=False)
        
        # Set up directory parameter callbacks
        self._setup_directory_callbacks()
        
        # Connect to parameter change signals
        self.params.sigTreeStateChanged.connect(self._on_param_changed)
        
        # Add tooltip handling using a simple approach
        try:
            # Install event filter on the parameter tree to catch mouse events
            self.tree.installEventFilter(self)
            print("Installed event filter on ParameterTree")
            
        except Exception as e:
            print(f"Could not install event filter: {e}")
            
        # Add a test for tooltip display using button clicks
        self.last_clicked_param = None
        
        left_layout.addWidget(QLabel("Config"))
        left_layout.addWidget(self.tree, stretch=3)

        # Preview Data button
        btn_preview = QPushButton("Preview Data")
        btn_preview.clicked.connect(self.preview_augmentation)
        left_layout.addWidget(btn_preview)

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
            self.gui_cfg = self.params_to_dict(self.params)
            self.append_log("Config updated from parameter tree")
            # Update other UI elements if needed
            self.apply_cfg_to_widgets()
            
            # Also try to show tooltip for the changed parameter
            # for change in changes:
            #     param_obj, change_type, data = change
            #     if change_type in ['value', 'expanded']:
            #         param_name = param_obj.name()
            #         self.last_clicked_param = param_name
                    # self._show_param_tooltip_simple(param_name)
                    # print(f"Parameter changed: {param_name} ({change_type})")
                    
        except Exception as e:
            self.append_log(f"Error updating config from parameters: {e}")
            self.append_log(f"Error updating config from parameters: {e}")
        
    def _on_tree_item_clicked_direct(self, item, column):
        """Handle direct tree widget item clicks to show tooltips."""
        try:
            # Try multiple ways to get parameter name from tree item
            param_name = None
            
            # Method 1: Try to get from itemMap
            if hasattr(self.tree, 'itemMap') and item in self.tree.itemMap:
                param_item = self.tree.itemMap[item]
                if hasattr(param_item, 'param'):
                    param_name = param_item.param.name()
            
            # Method 2: Try to get from item text
            if not param_name and item:
                param_name = item.text(0)  # Get text from first column
                
            # Method 3: Try to get from item data
            if not param_name and item:
                param_data = item.data(0, Qt.UserRole)
                if param_data:
                    param_name = str(param_data)
            
                
        except Exception as e:
            print(f"Error showing tooltip: {e}")
    
    def _on_tree_selection_changed(self):
        """Handle tree selection changes to show tooltips."""
        try:
            # Find selected items in the parameter tree
            tree_widgets = self.tree.findChildren(QTreeWidget)
            if tree_widgets:
                tree_widget = tree_widgets[0]
                current_item = tree_widget.currentItem()
                if current_item:
                    # Use the same logic as click handler
                    self._on_tree_item_clicked_direct(current_item, 0)
        except Exception as e:
            print(f"Error in selection changed: {e}")  # Debug print
        
    def eventFilter(self, obj, event):
        """Event filter to catch mouse clicks on the parameter tree."""
        if obj == self.tree and event.type() == event.Type.MouseButtonPress:
            try:
                # Get the position and try to find what was clicked
                pos = event.position().toPoint()
                print(f"Mouse clicked on ParameterTree at position: {pos}")
                
                # Try to use a timer to check selection after click
                QTimer.singleShot(100, self._delayed_tooltip_update)
                
            except Exception as e:
                print(f"Error in event filter: {e}")
        
        return super().eventFilter(obj, event)
    
    def _delayed_tooltip_update(self):
        """Update tooltip after a brief delay to allow selection to update."""
        try:
            self._refresh_tooltip_from_tree()
        except Exception as e:
            print(f"Error in delayed tooltip update: {e}")
    
    def _on_tree_view_clicked(self, index):
        """Handle QTreeView clicks."""
        try:
            if index.isValid():
                # Try to get parameter name from the index
                param_name = index.data()
                if param_name:
                    self._show_param_tooltip(param_name)
                    print(f"TreeView clicked: {param_name}")
        except Exception as e:
            print(f"Error in tree view clicked: {e}")
    
    def refresh_tree(self):
        """Update the ParameterTree with current comprehensive config data."""
        try:
            # Update comprehensive config from current gui_cfg
            self.sync_gui_to_comprehensive()
            
            # Recreate the parameter structure
            new_params = Parameter.create(**dict_to_params(self.comprehensive_cfg, "Configuration"))
            
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
    
    def sync_gui_to_comprehensive(self):
        """Sync the flat gui_cfg back to the comprehensive config structure."""
        try:
            # Update basic configuration
            if 'data' in self.gui_cfg:
                self.comprehensive_cfg['basic']['data'].update(self.gui_cfg['data'])
            if 'model' in self.gui_cfg:
                self.comprehensive_cfg['basic']['model'].update(self.gui_cfg['model'])
            if 'training' in self.gui_cfg:
                self.comprehensive_cfg['basic']['training'].update(self.gui_cfg['training'])
            if 'runtime' in self.gui_cfg:
                self.comprehensive_cfg['basic']['runtime'].update(self.gui_cfg['runtime'])
                
            # Update advanced configuration sections as needed
            for section in ['model_advanced', 'data_advanced', 'augmentation', 'training_advanced', 'evaluation', 'runtime_advanced']:
                if section in self.gui_cfg:
                    self.comprehensive_cfg['advanced'][section].update(self.gui_cfg[section])
                    
        except Exception as e:
            self.append_log(f"Error syncing to comprehensive config: {e}")
    
    def write_back_tree(self):
        """Extract data from the parameter tree back to both comprehensive and flat gui_cfg."""
        try:
            # Extract data from parameter tree
            tree_data = params_to_dict(self.params)
            
            # Update comprehensive config
            if 'basic' in tree_data:
                self.comprehensive_cfg['basic'] = tree_data['basic']
            if 'advanced' in tree_data:
                self.comprehensive_cfg['advanced'] = tree_data['advanced']
            
            # Flatten to gui_cfg for backward compatibility
            self.gui_cfg = {}
            if 'basic' in tree_data:
                self.gui_cfg.update(tree_data['basic'])
            if 'advanced' in tree_data:
                self.gui_cfg.update(tree_data['advanced'])
                
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
            if "runtime" not in self.gui_cfg:
                self.gui_cfg["runtime"] = {}
            self.gui_cfg["runtime"]["model_dir"] = d
            self.refresh_tree()

    # augment panel sync
    def apply_cfg_to_widgets(self):
        """Apply configuration to widgets (augmentation panel removed)."""
        # Note: Augmentation widgets have been removed from UI
        pass

    def sync_aug_to_cfg(self):
        """Sync augmentation configuration (widgets removed from UI)."""
        # Note: Augmentation widgets have been removed from UI
        # Configuration is now managed through the parameter tree
        pass
    
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
        path, _ = QFileDialog.getOpenFileName(self, "Choose image for preview")
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "Read failed", "Cannot read image.")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Get configuration from parameter tree instead of widgets
        self.write_back_tree()
        
        # Get image size from new config structure
        data_cfg = self.gui_cfg.get("data", {})
        image_size = data_cfg.get("image_size", [224, 224])
        if isinstance(image_size, list) and len(image_size) >= 2:
            size = int(image_size[0])
        else:
            size = 224
            
        # Use augmentation config from parameter tree
        aug_cfg = self.gui_cfg.get("augmentation", {})
        pipe = self.build_albu_pipeline(aug_cfg, size, size)
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
        dlg.setWindowTitle("Data Preview")
        layout = QGridLayout(dlg)
        pix = self.np_to_qpixmap(img_rgb)
        w0 = QLabel(); w0.setPixmap(pix.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel("Original"),0,0); layout.addWidget(w0,1,0)
        for i, s in enumerate(samples):
            p = self.np_to_qpixmap(s); l = QLabel(); l.setPixmap(p.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel(f"Aug {i+1}"),0,i+1); layout.addWidget(l,1,i+1)
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
        # sync GUI config from parameter tree
        self.write_back_tree()
        
        # Get model_dir from new config structure
        runtime_cfg = self.gui_cfg.get("runtime", {})
        model_dir = runtime_cfg.get("model_dir", "./model_dir")
        
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


    def create_comprehensive_config(self):
        """Create a comprehensive configuration structure with Basic and Advanced sections."""
        
        # Basic Configuration - Most commonly used parameters
        basic_config = {
            'data': {
                'train_dir': '',
                'val_dir': '',
                'image_size': [224, 224],
                'batch_size': 32,
                'num_classes': 1000,
                'shuffle': True
            },
            'model': {
                'backbone_type': 'resnet',
                'model_id': 50,
                'dropout_rate': 0.0,
                'activation': 'relu'
            },
            'training': {
                'epochs': 100,
                'learning_rate_type': 'exponential',
                'initial_learning_rate': 0.1,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'label_smoothing': 0.0
            },
            'runtime': {
                'model_dir': './model_dir',
                'distribution_strategy': 'mirrored',
                'mixed_precision': None,
                'num_gpus': 0
            }
        }
        
        # Advanced Configuration - Expert-level parameters
        advanced_config = {
            'model_advanced': {
                'depth_multiplier': 1.0,
                'stem_type': 'v0',
                'se_ratio': 0.0,
                'stochastic_depth_drop_rate': 0.0,
                'scale_stem': True,
                'resnetd_shortcut': False,
                'replace_stem_max_pool': False,
                'bn_trainable': True,
                'use_sync_bn': False,
                'norm_momentum': 0.99,
                'norm_epsilon': 0.001,
                'add_head_batch_norm': False,
                'kernel_initializer': 'random_uniform',
                'output_softmax': False
            },
            'data_advanced': {
                'tfds_name': '',
                'tfds_split': '',
                'cache': False,
                'shuffle_buffer_size': 10000,
                'cycle_length': 10,
                'block_length': 1,
                'drop_remainder': True,
                'sharding': True,
                'prefetch_buffer_size': None,
                'dtype': 'float32',
                'file_type': 'tfrecord',
                'image_field_key': 'image/encoded',
                'label_field_key': 'image/class/label',
                'decode_jpeg_only': True
            },
            'augmentation': {
                'type': 'augmentation_group',
                'name': 'augmentation'
            },
            'preprocessing': {
                'type': 'preprocessing_group', 
                'name': 'preprocessing'
            },
            'callbacks': {
                'type': 'callbacks_group',
                'name': 'callbacks'
            },
            'loss_functions': {
                'type': 'loss_functions_group',
                'name': 'loss_functions'
            },
            'metrics': {
                'type': 'metrics_group',
                'name': 'metrics'
            },
            'training_advanced': {
                'train_tf_while_loop': True,
                'train_tf_function': True,
                'eval_tf_function': True,
                'steps_per_loop': 1000,
                'summary_interval': 1000,
                'checkpoint_interval': 1000,
                'max_to_keep': 5,
                'validation_interval': 1000,
                'validation_steps': -1,
                'loss_upper_bound': 1000000.0,
                'one_hot_labels': True,
                'use_binary_cross_entropy': False,
                'soft_labels': False
            },
            'evaluation': {
                'top_k': 5,
                'report_per_class_metrics': False,
                'best_checkpoint_metric': '',
                'best_checkpoint_export_subdir': '',
                'best_checkpoint_metric_comp': 'higher'
            },
            'runtime_advanced': {
                'enable_xla': False,
                'run_eagerly': False,
                'per_gpu_thread_count': 0,
                'num_packs': 1,
                'loss_scale': None,
                'batchnorm_spatial_persistent': False,
                'tpu_settings': None,
                'all_reduce_alg': None
            }
        }
        
        return {
            'basic': basic_config,
            'advanced': advanced_config
        }

    def get_parameter_tooltip(self, param_name, section_name=None):
        """Get tooltip text for a parameter based on its name and section."""
        tooltips = {
            # Data section tooltips
            'train_dir': 'Path to the directory containing training data files (images, TFRecords, etc.)',
            'val_dir': 'Path to the directory containing validation/test data files',
            'image_size': 'Input image dimensions [width, height] - images will be resized to this size',
            'batch_size': 'Number of samples processed together in each training step. Larger values use more memory but may train faster',
            'num_classes': 'Number of different classes/categories in your dataset (e.g., 1000 for ImageNet)',
            'shuffle': 'Whether to randomly shuffle the training data order for each epoch',
            
            # Model section tooltips
            'backbone_type': 'The neural network architecture to use as the feature extractor (ResNet, EfficientNet, etc.)',
            'model_id': 'Specific variant of the backbone architecture (e.g., 50 for ResNet-50, 18 for ResNet-18)',
            'dropout_rate': 'Probability of randomly setting input units to 0 during training to prevent overfitting (0.0-1.0)',
            'activation': 'Activation function used in the neural network layers (ReLU, Swish, GELU, etc.)',
            
            # Training section tooltips
            'epochs': 'Number of complete passes through the entire training dataset',
            'learning_rate_type': 'Strategy for adjusting the learning rate during training (exponential decay, cosine annealing, etc.)',
            'initial_learning_rate': 'Starting learning rate value - how quickly the model learns from data',
            'momentum': 'SGD momentum factor - helps accelerate gradients in relevant directions (typically 0.9)',
            'weight_decay': 'L2 regularization strength to prevent overfitting by penalizing large weights',
            'label_smoothing': 'Technique to prevent overconfident predictions by softening target labels (0.0-0.3)',
            
            # Runtime section tooltips
            'model_dir': 'Directory where model checkpoints and training outputs will be saved',
            'distribution_strategy': 'Strategy for distributed training across multiple GPUs or machines',
            'mixed_precision': 'Use lower precision (float16/bfloat16) to speed up training and reduce memory usage',
            'num_gpus': 'Number of GPUs to use for training (0 for CPU-only)',
            
            # Model Advanced tooltips
            'depth_multiplier': 'Multiplier for the number of layers in the backbone network',
            'stem_type': 'Type of initial stem layers in the network architecture',
            'se_ratio': 'Squeeze-and-Excitation ratio for channel attention mechanisms',
            'stochastic_depth_drop_rate': 'Probability of dropping entire layers during training for regularization',
            'scale_stem': 'Whether to scale the stem layers in the network',
            'resnetd_shortcut': 'Use ResNet-D style shortcut connections for improved accuracy',
            'replace_stem_max_pool': 'Replace max pooling in stem with strided convolution',
            'bn_trainable': 'Whether batch normalization layers are trainable during fine-tuning',
            'use_sync_bn': 'Use synchronized batch normalization across multiple GPUs',
            'norm_momentum': 'Momentum for batch normalization moving average (typically 0.99)',
            'norm_epsilon': 'Small constant for numerical stability in batch normalization',
            'add_head_batch_norm': 'Add batch normalization before the final classification layer',
            'kernel_initializer': 'Method for initializing convolutional layer weights',
            'output_softmax': 'Apply softmax activation to final output (usually False for training)',
            
            # Data Advanced tooltips
            'tfds_name': 'TensorFlow Datasets name if using TFDS instead of custom data',
            'tfds_split': 'Which split of TFDS to use (train, validation, test)',
            'cache': 'Cache dataset in memory for faster access (requires sufficient RAM)',
            'shuffle_buffer_size': 'Size of buffer for shuffling data - larger values provide better randomness',
            'cycle_length': 'Number of input elements to process concurrently in parallel',
            'block_length': 'Number of consecutive elements from each input to read',
            'drop_remainder': 'Drop the last batch if it has fewer samples than batch_size',
            'sharding': 'Enable data sharding for distributed training',
            'prefetch_buffer_size': 'Number of batches to prefetch for pipeline optimization',
            'dtype': 'Data type for input tensors (float32 recommended for most cases)',
            'file_type': 'Format of input data files (TFRecord, SSTable, RecordIO)',
            'image_field_key': 'Key name for image data in TFRecord files',
            'label_field_key': 'Key name for label data in TFRecord files',
            'decode_jpeg_only': 'Only decode JPEG images, skip other formats for speed',
            
            # Augmentation tooltips
            'aug_rand_hflip': 'Randomly flip images horizontally during training',
            'aug_crop': 'Apply random cropping augmentation during training',
            'crop_area_range': 'Range of crop area as fraction of original image [min, max]',
            'center_crop_fraction': 'Fraction of image to keep when center cropping (for validation)',
            'color_jitter': 'Amount of random color variation (brightness, contrast, etc.)',
            'randaug_magnitude': 'Magnitude of RandAugment transformations (0-30)',
            'tf_resize_method': 'Method for resizing images (bilinear, nearest, bicubic, area)',
            'three_augment': 'Apply Three-Augment policy for advanced data augmentation',
            'is_multilabel': 'Whether this is a multi-label classification task',
            
            # Preprocessing tooltips
            'Resizing': 'Resize images to target dimensions with support for 2D and 3D data',
            'Normalization': 'Normalize data using various methods (min-max, zero-center, standardization, etc.)',
            'target_size': 'Target dimensions for resizing',
            'interpolation': 'Interpolation method for resizing',
            'preserve_aspect_ratio': 'Whether to preserve aspect ratio during resize',
            'data_format': 'Data format (2D for images, 3D for volumes)',
            'method': 'Normalization/processing method',
            'min_value': 'Minimum value for min-max normalization',
            'max_value': 'Maximum value for min-max normalization',
            'mean': 'Mean values for zero-center normalization',
            'std': 'Standard deviation values for standardization',
            'axis': 'Axis along which to normalize',
            'epsilon': 'Small constant to avoid division by zero',
            'r': 'Red channel value',
            'g': 'Green channel value', 
            'b': 'Blue channel value',
            
            # Callbacks tooltips
            'Early Stopping': 'Stop training when monitored metric stops improving',
            'Learning Rate Scheduler': 'Adjust learning rate during training based on metrics or schedule',
            'Model Checkpoint': 'Save model checkpoints during training',
            'CSV Logger': 'Log training metrics to CSV file',
            'TensorBoard': 'Log training metrics and model graph to TensorBoard',
            'monitor': 'Metric to monitor',
            'patience': 'Number of epochs with no improvement to wait',
            
            # Loss Functions tooltips  
            'loss_functions': 'Configure loss functions for training optimization',
            'Categorical Crossentropy': 'Standard categorical crossentropy loss for multi-class classification',
            'Sparse Categorical Crossentropy': 'Categorical crossentropy with integer labels (not one-hot)',
            'Binary Crossentropy': 'Binary crossentropy loss for binary classification',
            'Mean Squared Error': 'Mean squared error loss for regression tasks',
            'Mean Absolute Error': 'Mean absolute error loss for regression tasks',
            'Focal Loss': 'Focal loss for addressing class imbalance',
            'Huber Loss': 'Huber loss (smooth L1 loss) for robust regression',
            'from_logits': 'Whether predictions are logits or probabilities',
            'label_smoothing': 'Label smoothing factor to prevent overconfident predictions',
            'reduction': 'Type of reduction to apply to the loss',
            'alpha': 'Weighting factor for rare class (Focal Loss)',
            'gamma': 'Focusing parameter (Focal Loss)',
            'delta': 'Threshold at which to change between MSE and MAE (Huber Loss)',
            
            # Metrics tooltips
            'metrics': 'Configure metrics for training and evaluation monitoring',
            'Accuracy': 'Standard accuracy metric for classification tasks',
            'Categorical Accuracy': 'Categorical accuracy metric for multi-class classification',
            'Sparse Categorical Accuracy': 'Sparse categorical accuracy for integer label classification',
            'Top-K Categorical Accuracy': 'Top-K accuracy metric (e.g., top-5 accuracy)',
            'Precision': 'Precision metric for classification tasks',
            'Recall': 'Recall metric for classification tasks',
            'F1 Score': 'F1 score metric (harmonic mean of precision and recall)',
            'AUC': 'Area Under the Curve (AUC) metric',
            'name': 'Name for this metric in logs and outputs',
            'k': 'Number of top predictions to consider (Top-K accuracy)',
            'average': 'Averaging strategy for multi-class metrics',
            'class_id': 'Class ID for binary metrics (None for multiclass)',
            'curve': 'Curve type (ROC or Precision-Recall) for AUC',
            'multi_class': 'Multiclass strategy (one-vs-rest or one-vs-one) for AUC',
            'min_delta': 'Minimum change to qualify as improvement',
            'mode': 'Direction of improvement (min/max)',
            'restore_best_weights': 'Restore model weights from best epoch',
            'scheduler_type': 'Type of learning rate scheduler',
            'factor': 'Factor by which learning rate is reduced',
            'min_lr': 'Minimum learning rate',
            'step_size': 'Period of learning rate decay',
            'gamma': 'Multiplicative factor of learning rate decay',
            'filepath': 'Path template for checkpoint files',
            'save_best_only': 'Save only the best model',
            'save_weights_only': 'Save only model weights (not full model)',
            'period': 'Interval between checkpoints',
            'filename': 'Path to CSV log file',
            'separator': 'Delimiter for CSV file',
            'append': 'Append to existing file or create new',
            'log_dir': 'Directory for TensorBoard logs',
            'histogram_freq': 'Frequency for histogram computation',
            'write_graph': 'Write model graph to TensorBoard',
            'write_images': 'Write model weights as images',
            'update_freq': 'Update frequency for logging',
            
            # New augmentation method tooltips
            'Horizontal Flip': 'Randomly flip images horizontally for data augmentation',
            'Vertical Flip': 'Randomly flip images vertically for data augmentation',
            'Rotation': 'Randomly rotate images by specified angle range',
            'Gaussian Noise': 'Add random Gaussian noise to images for robustness',
            'Brightness Adjustment': 'Randomly adjust image brightness within specified limits',
            'Contrast Adjustment': 'Randomly adjust image contrast within specified limits',
            'Color Jittering': 'Randomly adjust hue, saturation, and value for color variations',
            'Random Cropping': 'Randomly crop portions of images with area and aspect ratio constraints',
            'enabled': 'Enable or disable this augmentation method',
            'probability': 'Probability of applying this augmentation (0.0 = never, 1.0 = always)',
            'angle_range': 'Maximum rotation angle in degrees (±)',
            'variance_limit': 'Maximum variance for Gaussian noise',
            'brightness_limit': 'Maximum brightness adjustment (±)',
            'contrast_limit': 'Maximum contrast adjustment (±)',
            'hue_shift_limit': 'Maximum hue shift in degrees',
            'sat_shift_limit': 'Maximum saturation shift percentage',
            'val_shift_limit': 'Maximum value/brightness shift percentage',
            'crop_area_min': 'Minimum crop area as fraction of original image',
            'crop_area_max': 'Maximum crop area as fraction of original image',
            'aspect_ratio_min': 'Minimum aspect ratio for cropped region',
            'aspect_ratio_max': 'Maximum aspect ratio for cropped region',
            
            # Custom augmentation tooltips
            'min_angle': 'Minimum rotation angle for custom rotation',
            'max_angle': 'Maximum rotation angle for custom rotation',
            'noise_type': 'Type of noise to add (gaussian, uniform, salt_pepper)',
            'intensity': 'Intensity of the noise effect',
            'blur_type': 'Type of blur to apply (gaussian, motion, median)',
            'blur_limit': 'Maximum blur kernel size',
            'distortion_type': 'Type of distortion to apply (elastic, perspective, barrel)',
            'distortion_strength': 'Strength of the distortion effect',
            'filter_type': 'Type of filter to apply (sharpen, emboss, edge_enhance)',
            'filter_strength': 'Strength of the filter effect',
            
            # Training Advanced tooltips
            'train_tf_while_loop': 'Use TensorFlow while loops for training (usually faster)',
            'train_tf_function': 'Use tf.function compilation for training loops',
            'eval_tf_function': 'Use tf.function compilation for evaluation loops',
            'steps_per_loop': 'Number of training steps per loop iteration',
            'summary_interval': 'How often to write training summaries (in steps)',
            'checkpoint_interval': 'How often to save model checkpoints (in steps)',
            'max_to_keep': 'Maximum number of recent checkpoints to keep',
            'validation_interval': 'How often to run validation evaluation (in steps)',
            'validation_steps': 'Number of validation steps per evaluation (-1 for full dataset)',
            'loss_upper_bound': 'Upper bound for loss values - training stops if exceeded',
            'one_hot_labels': 'Use one-hot encoded labels instead of sparse labels',
            'use_binary_cross_entropy': 'Use binary cross-entropy loss for binary classification',
            'soft_labels': 'Use soft labels instead of hard labels for training',
            
            # Evaluation tooltips
            'top_k': 'Compute top-K accuracy (e.g., top-5 accuracy for ImageNet)',
            'report_per_class_metrics': 'Report precision, recall, F1 for each class separately',
            'best_checkpoint_metric': 'Metric to use for selecting the best checkpoint',
            'best_checkpoint_export_subdir': 'Subdirectory to export the best checkpoint',
            'best_checkpoint_metric_comp': 'Whether higher or lower metric values are better',
            
            # Runtime Advanced tooltips
            'enable_xla': 'Enable XLA (Accelerated Linear Algebra) compilation for faster execution',
            'run_eagerly': 'Run in eager mode for debugging (disables optimizations)',
            'per_gpu_thread_count': 'Number of threads per GPU for data processing',
            'num_packs': 'Number of gradient packs for gradient compression',
            'loss_scale': 'Loss scaling factor for mixed precision training',
            'batchnorm_spatial_persistent': 'Use persistent batch normalization for spatial data',
            'tpu_settings': 'Special settings for TPU training',
            'all_reduce_alg': 'Algorithm for all-reduce operations in distributed training',
            
            # Special group parameter tooltips
            'width': 'Image width in pixels',
            'height': 'Image height in pixels',
            'min': 'Minimum value for the range',
            'max': 'Maximum value for the range',
        }
        
        # Section-specific tooltips for groups
        section_tooltips = {
            'basic': 'Essential parameters that most users need to configure',
            'advanced': 'Advanced parameters for expert users and fine-tuning',
            'data': 'Dataset and data loading configuration',
            'model': 'Neural network architecture settings',
            'training': 'Training process configuration',
            'runtime': 'Runtime and system configuration',
            'model_advanced': 'Advanced model architecture parameters',
            'data_advanced': 'Advanced data pipeline configuration',
            'augmentation': 'Data augmentation and preprocessing settings',
            'preprocessing': 'Data preprocessing methods including resizing and normalization',
            'callbacks': 'Training callbacks for monitoring, checkpointing, and scheduling',
            'training_advanced': 'Advanced training loop and optimization settings',
            'evaluation': 'Model evaluation and metrics configuration',
            'runtime_advanced': 'Advanced runtime and performance settings',
        }
        
        # Return tooltip based on parameter name or section
        if param_name in tooltips:
            return tooltips[param_name]
        elif param_name in section_tooltips:
            return section_tooltips[param_name]
        else:
            return f'Configuration parameter: {param_name}'

    def dict_to_params(self, data, name="Config"):
        """Convert a nested dictionary to Parameter tree structure with enhanced parameter types and tooltips."""
        if isinstance(data, dict):
            # Check if this is a special augmentation group type
            if data.get('type') == 'augmentation_group':
                return {
                    'name': data.get('name', name),
                    'type': 'augmentation_group',
                    'tip': self.get_parameter_tooltip('augmentation')
                }
            
            # Check if this is a special preprocessing group type
            if data.get('type') == 'preprocessing_group':
                return {
                    'name': data.get('name', name),
                    'type': 'preprocessing_group',
                    'tip': self.get_parameter_tooltip('preprocessing')
                }
            
            # Check if this is a special callbacks group type
            if data.get('type') == 'callbacks_group':
                return {
                    'name': data.get('name', name),
                    'type': 'callbacks_group',
                    'tip': self.get_parameter_tooltip('callbacks')
                }
            
            # Check if this is a special loss functions group type
            if data.get('type') == 'loss_functions_group':
                return {
                    'name': data.get('name', name),
                    'type': 'loss_functions_group',
                    'tip': self.get_parameter_tooltip('loss_functions')
                }
            
            # Check if this is a special metrics group type
            if data.get('type') == 'metrics_group':
                return {
                    'name': data.get('name', name),
                    'type': 'metrics_group',
                    'tip': self.get_parameter_tooltip('metrics')
                }
            
            children = []
            for key, value in data.items():
                if isinstance(value, dict):
                    # Nested dictionary - create a group
                    group_param = self.dict_to_params(value, key)
                    group_param['tip'] = self.get_parameter_tooltip(key)
                    children.append(group_param)
                elif isinstance(value, list):
                    # Handle list values
                    if len(value) == 2 and all(isinstance(x, int) for x in value):
                        # Image size or similar pair - create group with two int parameters
                        if key == 'image_size':
                            children.append({
                                'name': key,
                                'type': 'group',
                                'tip': self.get_parameter_tooltip(key),
                                'children': [
                                    {'name': 'width', 'type': 'int', 'value': value[0], 'limits': [1, 2048], 'tip': self.get_parameter_tooltip('width')},
                                    {'name': 'height', 'type': 'int', 'value': value[1], 'limits': [1, 2048], 'tip': self.get_parameter_tooltip('height')}
                                ]
                            })
                        elif key == 'crop_area_range':
                            children.append({
                                'name': key,
                                'type': 'group',
                                'tip': self.get_parameter_tooltip(key),
                                'children': [
                                    {'name': 'min', 'type': 'float', 'value': value[0], 'limits': [0.0, 1.0], 'step': 0.01, 'tip': self.get_parameter_tooltip('min')},
                                    {'name': 'max', 'type': 'float', 'value': value[1], 'limits': [0.0, 1.0], 'step': 0.01, 'tip': self.get_parameter_tooltip('max')}
                                ]
                            })
                        else:
                            # Generic two-element list
                            children.append({
                                'name': key,
                                'type': 'str',
                                'value': str(value),
                                'tip': self.get_parameter_tooltip(key)
                            })
                    else:
                        # Other lists - convert to string representation
                        children.append({
                            'name': key,
                            'type': 'str',
                            'value': str(value),
                            'tip': self.get_parameter_tooltip(key)
                        })
                else:
                    # Handle special directory/file parameters
                    if key in ['train_dir', 'val_dir'] and isinstance(value, str):
                        children.append({
                            'name': key,
                            'type': 'directory',
                            'value': value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key in ['model_dir'] and isinstance(value, str):
                        children.append({
                            'name': key,
                            'type': 'directory_only',
                            'value': value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    # Handle choice parameters
                    elif key == 'backbone_type':
                        values = ['resnet', 'efficientnet', 'mobilenet', 'vit', 'densenet']
                        # Ensure current value is valid, default to first item if not
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'activation':
                        values = ['relu', 'swish', 'gelu', 'leaky_relu', 'tanh']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'distribution_strategy':
                        values = ['mirrored', 'multi_worker_mirrored', 'tpu', 'parameter_server']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'learning_rate_type':
                        values = ['exponential', 'polynomial', 'cosine', 'constant', 'piecewise_constant']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'mixed_precision':
                        values = ['None', 'float16', 'bfloat16']  # Convert None to 'None' string for dropdown
                        # Handle None value properly
                        if value is None:
                            current_value = 'None'
                        else:
                            current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'optimizer_type':
                        values = ['sgd', 'adam', 'adamw', 'rmsprop', 'lars']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'file_type':
                        values = ['tfrecord', 'sstable', 'recordio']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'tf_resize_method':
                        values = ['bilinear', 'nearest', 'bicubic', 'area']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'kernel_initializer':
                        values = ['random_uniform', 'random_normal', 'glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'stem_type':
                        values = ['v0', 'v1', 'v2']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'dtype':
                        values = ['float32', 'float16', 'bfloat16']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key == 'best_checkpoint_metric_comp':
                        values = ['higher', 'lower']
                        current_value = value if value in values else values[0]
                        children.append({
                            'name': key,
                            'type': 'list',
                            'limits': values,
                            'value': current_value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    # Handle numeric parameters with appropriate ranges
                    elif key in ['batch_size', 'num_classes', 'epochs', 'model_id']:
                        limits = {
                            'batch_size': [1, 1024],
                            'num_classes': [1, 100000],
                            'epochs': [1, 1000],
                            'model_id': [18, 152]
                        }
                        children.append({
                            'name': key,
                            'type': 'int',
                            'value': int(value) if value else 0,
                            'limits': limits.get(key, [0, 1000000]),
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key in ['dropout_rate', 'learning_rate', 'initial_learning_rate', 'momentum', 'weight_decay', 'label_smoothing', 
                                'depth_multiplier', 'se_ratio', 'stochastic_depth_drop_rate', 'norm_momentum', 'norm_epsilon',
                                'color_jitter', 'center_crop_fraction']:
                        step = 0.01 if 'rate' in key or 'momentum' in key else 0.001
                        limits = [0.0, 1.0] if 'rate' in key or 'momentum' in key else [0.0, 10.0]
                        if key == 'initial_learning_rate':
                            limits = [1e-6, 1.0]
                            step = 0.001
                        elif key == 'norm_epsilon':
                            limits = [1e-8, 1e-3]
                            step = 1e-6
                        elif key == 'weight_decay':
                            limits = [0.0, 0.01]
                            step = 1e-5
                        
                        children.append({
                            'name': key,
                            'type': 'float',
                            'value': float(value) if value else 0.0,
                            'limits': limits,
                            'step': step,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    elif key in ['steps_per_loop', 'summary_interval', 'checkpoint_interval', 'validation_interval',
                                'shuffle_buffer_size', 'cycle_length', 'block_length', 'max_to_keep', 'validation_steps',
                                'top_k', 'randaug_magnitude', 'per_gpu_thread_count', 'num_packs', 'num_gpus']:
                        limits = {
                            'steps_per_loop': [1, 10000],
                            'summary_interval': [1, 10000],
                            'checkpoint_interval': [1, 10000],
                            'validation_interval': [1, 10000],
                            'shuffle_buffer_size': [1, 100000],
                            'cycle_length': [1, 100],
                            'block_length': [1, 100],
                            'max_to_keep': [1, 50],
                            'validation_steps': [-1, 10000],
                            'top_k': [1, 10],
                            'randaug_magnitude': [0, 30],
                            'per_gpu_thread_count': [0, 16],
                            'num_packs': [1, 8],
                            'num_gpus': [0, 8]
                        }
                        children.append({
                            'name': key,
                            'type': 'int',
                            'value': int(value) if value is not None else -1,
                            'limits': limits.get(key, [0, 100000]),
                            'tip': self.get_parameter_tooltip(key)
                        })
                    # Handle boolean parameters
                    elif isinstance(value, bool):
                        children.append({
                            'name': key,
                            'type': 'bool',
                            'value': value,
                            'tip': self.get_parameter_tooltip(key)
                        })
                    else:
                        # Default string parameter
                        children.append({
                            'name': key,
                            'type': 'str',
                            'value': str(value) if value is not None else '',
                            'tip': self.get_parameter_tooltip(key)
                        })
            
            return {
                'name': name,
                'type': 'group',
                'children': children,
                'tip': self.get_parameter_tooltip(name)
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
                'value': data,
                'tip': self.get_parameter_tooltip(name)
            }

    def params_to_dict(self, param):
        """Convert Parameter tree back to dictionary with proper handling of special cases."""
        if param.hasChildren():
            result = {}
            for child in param.children():
                child_name = child.name()
                
                # Special handling for augmentation group
                if isinstance(child, AugmentationGroup):
                    result[child_name] = self.extract_augmentation_config(child)
                elif isinstance(child, PreprocessingGroup):
                    result[child_name] = self.extract_preprocessing_config(child)
                elif isinstance(child, CallbacksGroup):
                    result[child_name] = self.extract_callbacks_config(child)
                elif child.hasChildren():
                    # Handle special group parameters like image_size or crop_area_range
                    if child_name == 'image_size' and len(child.children()) == 2:
                        width_child = next((c for c in child.children() if c.name() == 'width'), None)
                        height_child = next((c for c in child.children() if c.name() == 'height'), None)
                        if width_child and height_child:
                            result[child_name] = [width_child.value(), height_child.value()]
                        else:
                            result[child_name] = self.params_to_dict(child)
                    elif child_name == 'crop_area_range' and len(child.children()) == 2:
                        min_child = next((c for c in child.children() if c.name() == 'min'), None)
                        max_child = next((c for c in child.children() if c.name() == 'max'), None)
                        if min_child and max_child:
                            result[child_name] = [min_child.value(), max_child.value()]
                        else:
                            result[child_name] = self.params_to_dict(child)
                    else:
                        # Regular nested group
                        result[child_name] = self.params_to_dict(child)
                else:
                    # Leaf parameter
                    try:
                        value = child.value()
                    except Exception:
                        # Handle parameters without values set
                        value = None
                        
                    # Handle None values properly
                    if value == 'None' or value == '' or value is None:
                        # Check if this parameter should be None vs empty string
                        if child_name in ['mixed_precision', 'prefetch_buffer_size', 'tpu_settings', 'all_reduce_alg', 'loss_scale']:
                            result[child_name] = None
                        else:
                            result[child_name] = value
                    else:
                        result[child_name] = value
            return result
        else:
            return param.value()

    def extract_augmentation_config(self,aug_group):
        """Extract configuration from AugmentationGroup parameter."""
        config = {}
        
        for method_param in aug_group.children():
            method_name = method_param.name()
            method_config = {}
            
            # Extract configuration for each augmentation method
            for param in method_param.children():
                method_config[param.name()] = param.value()
            
            config[method_name] = method_config
        
        return config

    def extract_preprocessing_config(self,prep_group):
        """Extract configuration from PreprocessingGroup parameter."""
        config = {}
        
        for method_param in prep_group.children():
            method_name = method_param.name()
            
            # Skip the button
            if method_name == 'Load Custom Preprocessing':
                continue
                
            method_config = {}
            
            # Extract configuration for each preprocessing method
            for param in method_param.children():
                if param.hasChildren():
                    # Handle nested parameters (like target_size, mean, std)
                    nested_config = {}
                    for child_param in param.children():
                        nested_config[child_param.name()] = child_param.value()
                    method_config[param.name()] = nested_config
                else:
                    method_config[param.name()] = param.value()
            
            config[method_name] = method_config
        
        return config

    def extract_callbacks_config(self, callbacks_group):
        """Extract configuration from CallbacksGroup parameter."""
        config = {}
        
        for callback_param in callbacks_group.children():
            callback_name = callback_param.name()
            
            # Skip the button
            if callback_name == 'Load Custom Callbacks':
                continue
                
            callback_config = {}
            
            # Extract configuration for each callback
            for param in callback_param.children():
                callback_config[param.name()] = param.value()
            
            config[callback_name] = callback_config
        
        return config




    # helpers used above (np_to_qpixmap, build_albu_pipeline) - include simple definitions here:

    def create_custom_albumentations_transform(self, file_path, function_name, config):
        """Create a custom Albumentations transform from a Python function."""

        try:
            # Load the module from file
            spec = importlib.util.spec_from_file_location("custom_augmentations", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the function
            if not hasattr(module, function_name):
                print(f"Function {function_name} not found in {file_path}")
                return None
            
            custom_function = getattr(module, function_name)
            
            # Create a wrapper for Albumentations
            class CustomTransform(A.ImageOnlyTransform):
                def __init__(self, custom_func, func_config, always_apply=False, p=1.0):
                    super().__init__(always_apply, p)
                    self.custom_func = custom_func
                    self.func_config = func_config
                
                def apply(self, image, **params):
                    try:
                        # Extract parameters for the function (excluding enabled, file_path, function_name)
                        func_params = {}
                        for key, value in self.func_config.items():
                            if key not in ['enabled', 'file_path', 'function_name']:
                                func_params[key] = value
                        
                        # Call the custom function
                        result = self.custom_func(image, **func_params)
                        return result if result is not None else image
                    except Exception as e:
                        print(f"Error in custom augmentation function: {e}")
                        return image
            
            # Get probability from config
            probability = config.get('probability', 0.5)
            if 'p' in config:
                probability = config['p']
            
            return CustomTransform(custom_function, config, p=probability)
            
        except Exception as e:
            print(f"Error creating custom transform from {file_path}: {e}")
            return None

    def np_to_qpixmap(self,img: np.ndarray) -> QPixmap:
        if img is None:
            return QPixmap()
        if img.ndim == 2:
            h,w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h,w,c = img.shape
            if c == 3:
                qimg = QImage(img.data, w, h, w*3, QImage.Format.Format_RGB888)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if 'cv2' in globals() else img[:,:,0]
                h,w = gray.shape
                qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    def build_albu_pipeline(self, aug_cfg: Dict[str,Any], target_h:int, target_w:int):

        transforms = []
        
        # Handle new augmentation structure
        if isinstance(aug_cfg, dict):
            # Horizontal flip
            hflip = aug_cfg.get('Horizontal Flip', {})
            if hflip.get('enabled', False):
                prob = hflip.get('probability', 0.5)
                transforms.append(A.HorizontalFlip(p=prob))
            
            # Vertical flip
            vflip = aug_cfg.get('Vertical Flip', {})
            if vflip.get('enabled', False):
                prob = vflip.get('probability', 0.5)
                transforms.append(A.VerticalFlip(p=prob))
            
            # Rotation
            rotation = aug_cfg.get('Rotation', {})
            if rotation.get('enabled', False):
                angle = rotation.get('angle_range', 15.0)
                prob = rotation.get('probability', 0.5)
                transforms.append(A.Rotate(limit=angle, border_mode=cv2.BORDER_REFLECT_101, p=prob))
            
            # Gaussian noise
            noise = aug_cfg.get('Gaussian Noise', {})
            if noise.get('enabled', False):
                variance = noise.get('variance_limit', 0.01)
                prob = noise.get('probability', 0.2)
                transforms.append(A.GaussNoise(var_limit=(0, variance * 255**2), p=prob))
            
            # Brightness adjustment
            brightness = aug_cfg.get('Brightness Adjustment', {})
            if brightness.get('enabled', False):
                limit = brightness.get('brightness_limit', 0.2)
                prob = brightness.get('probability', 0.5)
                transforms.append(A.RandomBrightness(limit=limit, p=prob))
            
            # Contrast adjustment
            contrast = aug_cfg.get('Contrast Adjustment', {})
            if contrast.get('enabled', False):
                limit = contrast.get('contrast_limit', 0.2)
                prob = contrast.get('probability', 0.5)
                transforms.append(A.RandomContrast(limit=limit, p=prob))
            
            # Color jittering
            color_jitter = aug_cfg.get('Color Jittering', {})
            if color_jitter.get('enabled', False):
                hue = color_jitter.get('hue_shift_limit', 20)
                sat = color_jitter.get('sat_shift_limit', 30)
                val = color_jitter.get('val_shift_limit', 20)
                prob = color_jitter.get('probability', 0.5)
                transforms.append(A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=prob))
            
            # Random cropping
            crop = aug_cfg.get('Random Cropping', {})
            if crop.get('enabled', False):
                min_area = crop.get('crop_area_min', 0.08)
                max_area = crop.get('crop_area_max', 1.0)
                min_ratio = crop.get('aspect_ratio_min', 0.75)
                max_ratio = crop.get('aspect_ratio_max', 1.33)
                prob = crop.get('probability', 1.0)
                transforms.append(A.RandomResizedCrop(
                    height=target_h, width=target_w, 
                    scale=(min_area, max_area), 
                    ratio=(min_ratio, max_ratio), 
                    p=prob
                ))
            
            # Handle custom augmentations loaded from files
            for key, value in aug_cfg.items():
                if isinstance(value, dict) and value.get('enabled', False):
                    # Check if this is a custom function (has file_path and function_name)
                    if 'file_path' in value and 'function_name' in value:
                        try:
                            # Load and apply custom function
                            custom_transform = self.create_custom_albumentations_transform(
                                value['file_path'], 
                                value['function_name'], 
                                value
                            )
                            if custom_transform:
                                transforms.append(custom_transform)
                        except Exception as e:
                            print(f"Error loading custom augmentation {key}: {e}")
                    
                    # Handle legacy custom augmentations (without file_path)
                    elif 'Custom' in key:
                        try:
                            if 'Rotation' in key:
                                min_angle = value.get('min_angle', -45.0)
                                max_angle = value.get('max_angle', 45.0)
                                prob = value.get('probability', 0.5)
                                transforms.append(A.Rotate(limit=(min_angle, max_angle), p=prob))
                            
                            elif 'Noise' in key:
                                noise_type = value.get('noise_type', 'gaussian')
                                intensity = value.get('intensity', 0.05)
                                prob = value.get('probability', 0.3)
                                if noise_type == 'gaussian':
                                    transforms.append(A.GaussNoise(var_limit=(0, intensity * 255**2), p=prob))
                                elif noise_type == 'salt_pepper':
                                    transforms.append(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=prob))
                            
                            elif 'Blur' in key:
                                blur_type = value.get('blur_type', 'gaussian')
                                blur_limit = value.get('blur_limit', 3)
                                prob = value.get('probability', 0.2)
                                if blur_type == 'gaussian':
                                    transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=prob))
                                elif blur_type == 'motion':
                                    transforms.append(A.MotionBlur(blur_limit=(3, blur_limit), p=prob))
                                elif blur_type == 'median':
                                    transforms.append(A.MedianBlur(blur_limit=blur_limit, p=prob))
                            
                        except Exception as e:
                            print(f"Error adding legacy custom augmentation {key}: {e}")
        
        # Fallback to legacy structure
        else:
            if aug_cfg.get("aug_rand_hflip"):
                transforms.append(A.HorizontalFlip(p=0.5))
            if aug_cfg.get("three_augment"):
                transforms.append(A.VerticalFlip(p=0.5))
            
            # Use RandAugment magnitude instead of rotation limit
            randaug_mag = aug_cfg.get("randaug_magnitude", 0)
            if randaug_mag and randaug_mag > 0:
                transforms.append(A.Rotate(limit=min(randaug_mag, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.6))
            
            # Use crop area range
            crop_range = aug_cfg.get("crop_area_range", [0.08, 1.0])
            if isinstance(crop_range, list) and len(crop_range) >= 2:
                transforms.append(A.RandomResizedCrop(height=target_h, width=target_w, scale=(crop_range[0], crop_range[1]), ratio=(0.9,1.1), p=0.6))
            
            # Color jitter
            color_jitter = aug_cfg.get("color_jitter", 0.0)
            if color_jitter and color_jitter > 0:
                transforms.append(A.RandomBrightnessContrast(brightness_limit=color_jitter, contrast_limit=color_jitter, p=0.6))
        
        # Always add resize and normalize at the end
        if not any(isinstance(t, A.RandomResizedCrop) for t in transforms):
            transforms.append(A.Resize(target_h, target_w))
        transforms.append(A.Normalize())
        return A.Compose(transforms)
