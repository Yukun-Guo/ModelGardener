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
    QTreeWidget, QTabWidget, QScrollArea
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
from optimizer_group import OptimizerGroup
from data_loader_group import DataLoaderGroup
from config_manager import ConfigManager
from custom_functions_loader import CustomFunctionsLoader
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
pTypes.registerParameterType('optimizer_group', OptimizerGroup, override=True)
pTypes.registerParameterType('data_loader_group', DataLoaderGroup, override=True)


# ---------------------------
# MainWindow: similar UI as earlier, but start TFModelsTrainerThread
# ---------------------------
class MainWindow(QMainWindow):
    
    def __init__(self, experiment_name: str = "image_classification_imagenet"):
        super().__init__()
        self.setWindowTitle("🤖 ModelGardener - TensorFlow Training GUI")
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
        
        # Initialize enhanced configuration manager
        self.config_manager = ConfigManager(self)

        # left layout: config tree + augment controls + controls
        left_layout = QVBoxLayout()
        
        # Config buttons row (Load and Save)
        config_buttons_layout = QHBoxLayout()
        btn_load_config = QPushButton("Load Config")
        btn_load_config.clicked.connect(self.load_config)
        
        # Rename export package button to save configuration
        btn_save_configuration = QPushButton("Save Configuration")
        btn_save_configuration.clicked.connect(self.export_shareable_package)
        btn_save_configuration.setToolTip("Save configuration with embedded custom functions")
        
        # Apply consistent styling to config buttons
        button_style = """
            QPushButton {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 11pt;
                font-weight: 500;
                padding: 8px 16px;
                background-color: #7f8c8d;
                color: white;
                border: none;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #6c7b7c;
            }
            QPushButton:pressed {
                background-color: #5d6a6b;
            }
        """
        
        # Auto-reload button gets a different color to distinguish it
        save_config_style = """
            QPushButton {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 11pt;
                font-weight: 500;
                padding: 8px 16px;
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """
        
        btn_load_config.setStyleSheet(button_style)
        btn_save_configuration.setStyleSheet(save_config_style)
        
        config_buttons_layout.addWidget(btn_load_config)
        config_buttons_layout.addWidget(btn_save_configuration)
        config_buttons_layout.addStretch()  # Add stretch to push buttons to left
        
        # Store reference for later use
        self.last_loaded_custom_functions = None
        
        left_layout.addLayout(config_buttons_layout)
        
        # Create ParameterTree with comprehensive config data organized in Basic/Advanced sections
        self.params = Parameter.create(**self.dict_to_params(self.comprehensive_cfg, "Configuration"))
        self.tree = ParameterTree()
        self.tree.setParameters(self.params, showTop=False)
        
        # Apply professional styling to the parameter tree
        #self._apply_parameter_tree_styling()
        
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
        
        # Create a styled configuration label
        config_label = QLabel("Configuration")
        config_label.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 14pt;
                font-weight: 600;
                color: #fcfcfc;
                padding: 8px 0px;
                background-color: transparent;
            }
        """)
        
        left_layout.addWidget(config_label)
        left_layout.addWidget(self.tree, stretch=3)

        # Control panel below the tree
        control_panel_layout = QVBoxLayout()
        
        # Data preview button
        btn_preview = QPushButton("Preview Data")
        btn_preview.clicked.connect(self.preview_augmentation)
        
        # Apply consistent styling to control buttons
        control_button_style = """
            QPushButton {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12pt;
                font-weight: 600;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-height: 35px;
            }
        """
        
        # Preview button with subtle accent
        preview_button_style = control_button_style + """
            QPushButton {
                background-color: #5d6d7e;
                color: white;
            }
            QPushButton:hover {
                background-color: #4a5a6b;
            }
            QPushButton:pressed {
                background-color: #3e4a57;
            }
        """
        btn_preview.setStyleSheet(preview_button_style)
        control_panel_layout.addWidget(btn_preview)
        
        # Training control buttons
        training_buttons_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start Training")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("⏹ Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        
        # Style start button with subtle success accent
        start_button_style = control_button_style + """
            QPushButton {
                background-color: #52c41a;
                color: white;
            }
            QPushButton:hover {
                background-color: #46b317;
            }
            QPushButton:pressed {
                background-color: #3a9614;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #bdc3c7;
            }
        """
        
        # Style stop button with subtle warning accent
        stop_button_style = control_button_style + """
            QPushButton {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #bdc3c7;
            }
        """
        
        self.btn_start.setStyleSheet(start_button_style)
        self.btn_stop.setStyleSheet(stop_button_style)
        
        training_buttons_layout.addWidget(self.btn_start)
        training_buttons_layout.addWidget(self.btn_stop)
        control_panel_layout.addLayout(training_buttons_layout)
        
        # Checkpoint and model directory buttons
        path_buttons_layout = QHBoxLayout()
        btn_ckpt = QPushButton("📁 Choose Checkpoint")
        btn_ckpt.clicked.connect(self.choose_checkpoint)
        btn_model_dir = QPushButton("📂 Choose Model Dir")
        btn_model_dir.clicked.connect(self.choose_model_dir)
        
        # Style path buttons with consistent neutral design
        path_button_style = control_button_style + """
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                font-size: 11pt;
                padding: 8px 16px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #6c7b7c;
            }
            QPushButton:pressed {
                background-color: #5d6a6b;
            }
        """
        btn_ckpt.setStyleSheet(path_button_style)
        btn_model_dir.setStyleSheet(path_button_style)
        
        path_buttons_layout.addWidget(btn_ckpt)
        path_buttons_layout.addWidget(btn_model_dir)
        control_panel_layout.addLayout(path_buttons_layout)
        
        # Progress bar with consistent styling
        self.progress = QProgressBar()
        progress_style = """
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                text-align: center;
                font-size: 10pt;
                font-weight: 600;
                color: #2c3e50;
                background-color: #ecf0f1;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """
        self.progress.setStyleSheet(progress_style)
        control_panel_layout.addWidget(self.progress)
        
        left_layout.addLayout(control_panel_layout)

        left_widget = QWidget(); left_widget.setLayout(left_layout)

        # right layout: Tab widget with multiple tabs
        right_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        
        # TensorBoard tab
        tensorboard_tab = QWidget()
        tensorboard_layout = QVBoxLayout()
        self.tb_view = QWebEngineView()
        self.tb_view.setUrl(f"http://localhost:6006")  # Default URL for TensorBoard
        tensorboard_layout.addWidget(self.tb_view)
        tensorboard_tab.setLayout(tensorboard_layout)
        self.tab_widget.addTab(tensorboard_tab, "TensorBoard")
        
        # Data Preview tab
        data_preview_tab = QWidget()
        data_preview_layout = QVBoxLayout()
        self.preview_label = QLabel("Click 'Preview Data' in the control panel to see data samples here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(True)
        data_preview_layout.addWidget(self.preview_scroll)
        data_preview_tab.setLayout(data_preview_layout)
        self.tab_widget.addTab(data_preview_tab, "Data Preview")
        
        # Testing tab
        testing_tab = QWidget()
        testing_layout = QVBoxLayout()
        testing_layout.addWidget(QLabel("Testing Tools"))
        
        # Add model testing controls
        test_controls_layout = QHBoxLayout()
        btn_test_model = QPushButton("Test Current Model")
        btn_load_test_data = QPushButton("Load Test Dataset")
        btn_run_evaluation = QPushButton("Run Evaluation")
        
        test_controls_layout.addWidget(btn_test_model)
        test_controls_layout.addWidget(btn_load_test_data)
        test_controls_layout.addWidget(btn_run_evaluation)
        testing_layout.addLayout(test_controls_layout)
        
        # Test results area
        self.test_results = QPlainTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setPlaceholderText("Test results will appear here...")
        testing_layout.addWidget(QLabel("Test Results:"))
        testing_layout.addWidget(self.test_results)
        
        testing_tab.setLayout(testing_layout)
        self.tab_widget.addTab(testing_tab, "Testing")
        
        # Deploy tab
        deploy_tab = QWidget()
        deploy_layout = QVBoxLayout()
        deploy_layout.addWidget(QLabel("Model Deployment"))
        
        # Deployment controls
        deploy_controls_layout = QVBoxLayout()
        
        export_controls_layout = QHBoxLayout()
        btn_export_savedmodel = QPushButton("Export SavedModel")
        btn_export_tflite = QPushButton("Export TFLite")
        btn_export_onnx = QPushButton("Export ONNX")
        
        export_controls_layout.addWidget(btn_export_savedmodel)
        export_controls_layout.addWidget(btn_export_tflite)
        export_controls_layout.addWidget(btn_export_onnx)
        deploy_controls_layout.addLayout(export_controls_layout)
        
        deployment_controls_layout = QHBoxLayout()
        btn_deploy_local = QPushButton("Deploy Locally")
        btn_deploy_cloud = QPushButton("Deploy to Cloud")
        btn_create_container = QPushButton("Create Docker Container")
        
        deployment_controls_layout.addWidget(btn_deploy_local)
        deployment_controls_layout.addWidget(btn_deploy_cloud)
        deployment_controls_layout.addWidget(btn_create_container)
        deploy_controls_layout.addLayout(deployment_controls_layout)
        
        deploy_layout.addLayout(deploy_controls_layout)
        
        # Deployment status/logs
        self.deploy_log = QPlainTextEdit()
        self.deploy_log.setReadOnly(True)
        self.deploy_log.setPlaceholderText("Deployment logs will appear here...")
        deploy_layout.addWidget(QLabel("Deployment Logs:"))
        deploy_layout.addWidget(self.deploy_log)
        
        deploy_tab.setLayout(deploy_layout)
        self.tab_widget.addTab(deploy_tab, "Deploy")
        
        # Prediction tab
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(QLabel("Model Prediction"))
        
        # Model loading controls
        model_load_layout = QHBoxLayout()
        btn_load_model = QPushButton("Load Deployed Model")
        self.model_path_label = QLabel("No model loaded")
        model_load_layout.addWidget(btn_load_model)
        model_load_layout.addWidget(self.model_path_label)
        prediction_layout.addLayout(model_load_layout)
        
        # Prediction controls
        pred_controls_layout = QHBoxLayout()
        btn_select_image = QPushButton("Select Image")
        btn_predict = QPushButton("Run Prediction")
        btn_batch_predict = QPushButton("Batch Prediction")
        
        pred_controls_layout.addWidget(btn_select_image)
        pred_controls_layout.addWidget(btn_predict)
        pred_controls_layout.addWidget(btn_batch_predict)
        prediction_layout.addLayout(pred_controls_layout)
        
        # Prediction results area
        pred_results_layout = QHBoxLayout()
        
        # Image preview
        self.pred_image_label = QLabel("No image selected")
        self.pred_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_image_label.setMinimumSize(300, 300)
        self.pred_image_label.setStyleSheet("border: 1px solid gray;")
        pred_results_layout.addWidget(self.pred_image_label)
        
        # Prediction results
        pred_text_layout = QVBoxLayout()
        pred_text_layout.addWidget(QLabel("Prediction Results:"))
        self.pred_results = QPlainTextEdit()
        self.pred_results.setReadOnly(True)
        self.pred_results.setPlaceholderText("Prediction results will appear here...")
        pred_text_layout.addWidget(self.pred_results)
        pred_results_layout.addLayout(pred_text_layout)
        
        prediction_layout.addLayout(pred_results_layout)
        
        prediction_tab.setLayout(prediction_layout)
        self.tab_widget.addTab(prediction_tab, "Prediction")
        
        # Logs tab (moved from being standalone to being a tab)
        logs_tab = QWidget()
        logs_layout = QVBoxLayout()
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        logs_layout.addWidget(QLabel("Training Logs"))
        logs_layout.addWidget(self.log_edit)
        logs_tab.setLayout(logs_layout)
        self.tab_widget.addTab(logs_tab, "Logs")
        
        right_layout.addWidget(self.tab_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # Connect button handlers for new UI elements
        self._connect_tab_handlers()

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
    
    def _apply_parameter_tree_styling(self):
        """Apply professional styling to the parameter tree with consistent color scheme."""
        try:
            # Set custom stylesheet for the parameter tree
            style_sheet = """
            QTreeWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12pt;
                background-color: #f7f8fc;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                selection-background-color: #3498db;
                selection-color: white;
                outline: none;
            }
            
            QTreeWidget::item {
                padding: 6px;
                margin: 2px;
                height: 30px;
                border-bottom: 1px solid #ecf0f1;
                color: #2c3e50;
                font-weight: 500;
            }
            
            QTreeWidget::item:selected {
                background-color: #3498db;
                color: white;
                border: none;
            }
            
            QTreeWidget::item:hover {
                background-color: #ebf3fd;
                color: #2980b9;
            }
            
            QTreeWidget::item:has-children {
                font-weight: 600;
                color: #34495e;
                background-color: #eaebec;
            }
            
            QTreeWidget::item:has-children:selected {
                background-color: #2980b9;
                color: white;
                font-weight: 600;
            }
            
            QTreeWidget::branch:has-siblings:!adjoins-item {
                border-image: none;
                image: none;
            }
            
            QTreeWidget::branch:has-siblings:adjoins-item {
                border-image: none;
                image: none;
            }
            
            QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
                border-image: none;
                image: none;
            }
            
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTYgNEwxMCA4TDYgMTJWNFoiIGZpbGw9IiM3Zjhj8WQiLz4KPHN2Zz4K);
            }
            
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                border-image: none;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQgNkw4IDEwTDEyIDZINFoiIGZpbGw9IiM3Zjhj8WQiLz4KPHN2Zz4K);
            }
            
            /* Style for parameter value editors */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                font-size: 11pt;
                padding: 4px 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #3498db;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #3498db;
                outline: none;
                box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
            }
            
            QCheckBox {
                font-size: 11pt;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #3498db;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjg1NCAzLjY0NkwxNC44NTQgNC42NDZMMTQuMTQ2IDUuMzU0TDYuNSAxMy4wVjEzSDZMMi44NTQgOS44NTRMMC44NTQgMTEuODU0TDcuNSA0LjVMMTMuODU0IDMuNjQ2WiIgZmlsbD0id2hpdGUiLz4KPHN2Zz4K);
            }
            
            QCheckBox::indicator:hover {
                border-color: #3498db;
            }
            
            QPushButton {
                font-size: 11pt;
                padding: 6px 12px;
                background-color: #7f8c8d;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #6c7b7c;
            }
            
            QPushButton:pressed {
                background-color: #5d6a6b;
            }
            """
            
            self.tree.setStyleSheet(style_sheet)
            
            # Set additional properties for better appearance
            self.tree.setAlternatingRowColors(False)
            self.tree.setRootIsDecorated(True)
            self.tree.setIndentation(20)
            self.tree.setHeaderHidden(False)
            
            # Set header styling
            header = self.tree.header()
            if header:
                header.setStyleSheet("""
                    QHeaderView::section {
                        background-color: #34495e;
                        color: white;
                        padding: 8px;
                        border: none;
                        font-size: 12pt;
                        font-weight: 600;
                    }
                """)
                header.setDefaultSectionSize(200)
                header.setStretchLastSection(True)
            
        except Exception as e:
            print(f"Error applying parameter tree styling: {e}")
    
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
            new_params = Parameter.create(**self.dict_to_params(self.comprehensive_cfg, "Configuration"))
            
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
            for section in ['model_advanced', 'data_advanced', 'augmentation', 'callbacks', 'training_advanced', 'evaluation', 'runtime_advanced']:
                if section in self.gui_cfg:
                    self.comprehensive_cfg['advanced'][section].update(self.gui_cfg[section])
                    
        except Exception as e:
            self.append_log(f"Error syncing to comprehensive config: {e}")
    
    def write_back_tree(self):
        """Extract data from the parameter tree back to both comprehensive and flat gui_cfg."""
        try:
            # Extract data from parameter tree
            tree_data = self.params_to_dict(self.params)
            
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
    def load_config(self):
        """Enhanced load configuration with custom functions support."""
        path, _ = QFileDialog.getOpenFileName(self, "Load config", filter="*.json *.yaml *.yml")
        if not path:
            return
            
        try:
            # Load enhanced configuration
            config_data, custom_functions_info = self.config_manager.load_enhanced_config(path)
            
            if config_data is None:
                return  # Error already shown by config_manager
            
            # Update configuration
            self.gui_cfg = config_data
            
            # Store original configuration for custom groups (before apply_cfg_to_widgets modifies it)
            self.original_gui_cfg = config_data.copy()
            
            # Load custom functions FIRST if they exist, before applying config
            custom_count = sum(len(funcs) for funcs in (custom_functions_info or {}).values())
            if custom_count > 0:
                # Ask user if they want to auto-reload custom functions
                info_msg = f"Loaded: {path}"
                info_msg += f"\n\nFound {custom_count} custom function(s) to reload:"
                for func_type, funcs in custom_functions_info.items():
                    if funcs:
                        info_msg += f"\n- {func_type.replace('_', ' ').title()}: {len(funcs)}"
                
                info_msg += "\n\nCustom functions must be loaded before applying configuration."
                info_msg += "\nAuto-reload custom functions now?"
                
                reply = QMessageBox.question(self, "Configuration Loaded", info_msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes)
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Load custom functions FIRST, before applying configuration
                    self.auto_reload_custom_functions(custom_functions_info)
                    
                    # Now apply the configuration with custom functions available
                    # DO NOT call refresh_tree() as it recreates the parameter tree and loses custom functions
                    self.apply_cfg_to_widgets()
                    
                    # Apply configuration to custom parameter groups
                    self._apply_config_to_custom_groups()
                else:
                    # User chose not to reload custom functions
                    self.apply_cfg_to_widgets()
                    self.refresh_tree()
                    QMessageBox.warning(self, "Warning", 
                        "Configuration loaded without custom functions. "
                        "Custom function selections may not work correctly until functions are loaded.")
            else:
                # No custom functions, proceed normally
                self.apply_cfg_to_widgets()
                self.refresh_tree()
                QMessageBox.information(self, "Loaded", f"Loaded: {path}")
                
            # Store custom functions info for later auto-reload
            self.last_loaded_custom_functions = custom_functions_info
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load configuration:\n{str(e)}")
    
    def _apply_config_to_custom_groups(self):
        """Apply configuration to custom parameter groups after custom functions are loaded."""
        try:
            # Apply configuration to data loader group
            basic_group = self.params.child('basic')
            if basic_group:
                data_group = basic_group.child('data')
                if data_group:
                    data_loader_group = data_group.child('data_loader')
                    if data_loader_group and hasattr(data_loader_group, 'set_data_loader_config'):
                        # Get data loader config from ORIGINAL gui_cfg (not modified by apply_cfg_to_widgets)
                        original_basic_config = self.original_gui_cfg.get('basic', {})
                        original_data_config = original_basic_config.get('data', {})
                        original_data_loader_config = original_data_config.get('data_loader', {})
                        
                        if original_data_loader_config:
                            data_loader_group.set_data_loader_config(original_data_loader_config)
                            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.append_log(f"Error applying config to custom groups: {e}")
    
    def auto_reload_custom_functions(self, custom_functions_info: Dict[str, Any]):
        """Automatically reload custom functions from metadata."""
        if not custom_functions_info:
            return
            
        try:
            reload_results = []
            total_attempted = 0
            total_successful = 0
            
            # First, make sure we have a parameter tree
            if not hasattr(self, 'params') or not self.params:
                self.refresh_tree()
            
            # Reload data loaders
            for loader_info in custom_functions_info.get('data_loaders', []):
                total_attempted += 1
                
                try:
                    # Find the data loader group
                    basic_group = self.params.child('basic')
                    data_group = basic_group.child('data') if basic_group else None
                    data_loader_group = data_group.child('data_loader') if data_group else None
                    
                    if data_loader_group and hasattr(data_loader_group, 'load_custom_data_loader_from_metadata'):
                        success = data_loader_group.load_custom_data_loader_from_metadata(loader_info)
                        if success:
                            reload_results.append(f"✓ Data loader: {loader_info['name']}")
                            total_successful += 1
                        else:
                            reload_results.append(f"✗ Data loader failed: {loader_info['name']}")
                    else:
                        # Fallback to the old method
                        file_path = loader_info['file_path']
                        function_name = loader_info['original_name']
                        
                        if os.path.exists(file_path) and data_loader_group:
                            success = CustomFunctionsLoader.load_custom_data_loader_from_file(
                                data_loader_group, file_path, function_name
                            )
                            if success:
                                reload_results.append(f"✓ Data loader: {loader_info['name']}")
                                total_successful += 1
                            else:
                                reload_results.append(f"✗ Data loader failed: {loader_info['name']}")
                        else:
                            reload_results.append(f"✗ Data loader file not found or group not accessible: {loader_info['name']}")
                            
                except Exception as e:
                    reload_results.append(f"✗ Data loader error: {loader_info['name']} - {e}")
            
            # Reload loss functions
            for loss_info in custom_functions_info.get('loss_functions', []):
                total_attempted += 1
                file_path = loss_info['file_path']
                function_name = loss_info['function_name']
                
                if os.path.exists(file_path):
                    try:
                        # Find the loss functions group
                        basic_group = self.params.child('basic')
                        model_group = basic_group.child('model') if basic_group else None
                        loss_group = model_group.child('loss_functions') if model_group else None
                        
                        if loss_group:
                            success = CustomFunctionsLoader.load_custom_loss_function_from_file(
                                loss_group, file_path, function_name
                            )
                            if success:
                                reload_results.append(f"✓ Loss function: {loss_info['name']}")
                                total_successful += 1
                            else:
                                reload_results.append(f"✗ Loss function failed: {loss_info['name']}")
                        else:
                            reload_results.append(f"✗ Loss function group not accessible: {loss_info['name']}")
                    except Exception as e:
                        reload_results.append(f"✗ Loss function error: {loss_info['name']} - {e}")
                else:
                    reload_results.append(f"✗ Loss function file not found: {file_path}")
            
            # Reload augmentations
            for aug_info in custom_functions_info.get('augmentations', []):
                total_attempted += 1
                file_path = aug_info['file_path']
                function_name = aug_info['function_name']
                
                if os.path.exists(file_path):
                    try:
                        # Find the augmentation group
                        advanced_group = self.params.child('advanced')
                        aug_group = advanced_group.child('augmentation') if advanced_group else None
                        
                        if aug_group:
                            success = CustomFunctionsLoader.load_custom_augmentation_from_file(
                                aug_group, file_path, function_name
                            )
                            if success:
                                reload_results.append(f"✓ Augmentation: {aug_info['name']}")
                                total_successful += 1
                            else:
                                reload_results.append(f"✗ Augmentation failed: {aug_info['name']}")
                        else:
                            reload_results.append(f"✗ Augmentation group not accessible: {aug_info['name']}")
                    except Exception as e:
                        reload_results.append(f"✗ Augmentation error: {aug_info['name']} - {e}")
                else:
                    reload_results.append(f"✗ Augmentation file not found: {file_path}")
            
            # Reload callbacks
            for callback_info in custom_functions_info.get('callbacks', []):
                total_attempted += 1
                file_path = callback_info['file_path']
                function_name = callback_info['function_name']
                
                if os.path.exists(file_path):
                    try:
                        # Find the callbacks group
                        advanced_group = self.params.child('advanced')
                        callback_group = advanced_group.child('callbacks') if advanced_group else None
                        
                        if callback_group:
                            success = CustomFunctionsLoader.load_custom_callback_from_file(
                                callback_group, file_path, function_name
                            )
                            if success:
                                reload_results.append(f"✓ Callback: {callback_info['name']}")
                                total_successful += 1
                            else:
                                reload_results.append(f"✗ Callback failed: {callback_info['name']}")
                        else:
                            reload_results.append(f"✗ Callback group not accessible: {callback_info['name']}")
                    except Exception as e:
                        reload_results.append(f"✗ Callback error: {callback_info['name']} - {e}")
                else:
                    reload_results.append(f"✗ Callback file not found: {file_path}")
            
            # Reload preprocessing
            for prep_info in custom_functions_info.get('preprocessing', []):
                total_attempted += 1
                file_path = prep_info['file_path']
                function_name = prep_info['function_name']
                
                if os.path.exists(file_path):
                    try:
                        # Find the preprocessing group
                        basic_group = self.params.child('basic')
                        data_group = basic_group.child('data') if basic_group else None
                        prep_group = data_group.child('preprocessing') if data_group else None
                        
                        if prep_group:
                            success = CustomFunctionsLoader.load_custom_preprocessing_from_file(
                                prep_group, file_path, function_name
                            )
                            if success:
                                reload_results.append(f"✓ Preprocessing: {prep_info['name']}")
                                total_successful += 1
                            else:
                                reload_results.append(f"✗ Preprocessing failed: {prep_info['name']}")
                        else:
                            reload_results.append(f"✗ Preprocessing group not accessible: {prep_info['name']}")
                    except Exception as e:
                        reload_results.append(f"✗ Preprocessing error: {prep_info['name']} - {e}")
                else:
                    reload_results.append(f"✗ Preprocessing file not found: {file_path}")
            
            # Reload optimizers
            for opt_info in custom_functions_info.get('optimizers', []):
                total_attempted += 1
                function_name = opt_info['function_name']
                
                try:
                    # Find the optimizer group
                    basic_group = self.params.child('basic')
                    model_group = basic_group.child('model') if basic_group else None
                    optimizer_group = model_group.child('optimizer') if model_group else None
                    
                    if optimizer_group:
                        # Note: For optimizers, we need the file path which might not be stored
                        # This is a limitation of the current optimizer storage system
                        reload_results.append(f"⚠ Optimizer: {opt_info['name']} (needs manual reload)")
                    else:
                        reload_results.append(f"✗ Optimizer group not accessible: {opt_info['name']}")
                except Exception as e:
                    reload_results.append(f"✗ Optimizer error: {opt_info['name']} - {e}")
            
            # Show results
            if reload_results:
                result_msg = f"Custom Functions Auto-Reload Results ({total_successful}/{total_attempted} successful):\n\n"
                result_msg += "\n".join(reload_results)
                
                if total_successful == total_attempted:
                    QMessageBox.information(self, "Auto-Reload Complete", result_msg)
                else:
                    result_msg += "\n\nSome functions failed to reload. You may need to:"
                    result_msg += "\n• Check if the source files still exist"
                    result_msg += "\n• Manually reload failed functions using 'Load Custom...' buttons"
                    result_msg += "\n• Update file paths if they have changed"
                    QMessageBox.warning(self, "Auto-Reload Partial Success", result_msg)
            else:
                QMessageBox.information(self, "Auto-Reload", 
                    "No custom functions were found to reload.")
                
        except Exception as e:
            QMessageBox.critical(self, "Auto-Reload Error", 
                f"Failed to auto-reload custom functions:\n{str(e)}")

    def export_shareable_package(self):
        """Save configuration with embedded custom functions and create shareable package."""
        try:
            # Sync current configuration
            self.sync_aug_to_cfg()
            
            # Get directory to save the package
            package_dir = QFileDialog.getExistingDirectory(
                self, 
                "Select Directory to Save Configuration",
                ""
            )
            
            if not package_dir:
                return
            
            # Create a subdirectory with timestamp for the package
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"ModelGardener_Config_Package_{timestamp}"
            full_package_path = os.path.join(package_dir, package_name)
            
            # Collect custom functions information
            custom_functions_info = None
            if hasattr(self, 'params'):
                custom_functions_info = self.config_manager.collect_custom_functions_info(self.params)
            
            # Check if there are any custom functions
            custom_count = sum(len(funcs) for funcs in (custom_functions_info or {}).values()) if custom_functions_info else 0
            
            if custom_count == 0:
                reply = QMessageBox.question(
                    self, "No Custom Functions",
                    "No custom functions were found in the current configuration.\n\n"
                    "Do you want to create a package with just the base configuration?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply != QMessageBox.StandardButton.Yes:
                    return
            
            # Create the package
            success = self.config_manager.create_shareable_package(
                self.gui_cfg,
                full_package_path,
                custom_functions_info,
                include_readme=True
            )
            
            if success:
                # Show success message with details
                message = f"Configuration saved successfully!\n\n"
                message += f"Location: {full_package_path}\n\n"
                message += "Package contents:\n"
                message += "• model_config.json - Configuration file\n"
                
                if custom_count > 0:
                    message += f"• custom_functions/ - {custom_count} custom function file(s)\n"
                    message += "• custom_functions_manifest.json - Function metadata\n"
                    message += "• setup_custom_functions.py - Setup script\n"
                
                message += "• README.md - Setup instructions\n\n"
                message += "This configuration can be shared with others and easily imported into ModelGardener."
                
                QMessageBox.information(self, "Configuration Saved", message)
                
                # Ask if user wants to open the configuration directory
                reply = QMessageBox.question(
                    self, "Open Configuration Directory",
                    "Would you like to open the configuration directory?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        # Open file explorer (works on Windows, macOS, and most Linux distros)
                        import subprocess
                        import platform
                        
                        if platform.system() == "Windows":
                            subprocess.run(["explorer", full_package_path])
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", full_package_path])
                        else:  # Linux and others
                            subprocess.run(["xdg-open", full_package_path])
                    except Exception as e:
                        QMessageBox.information(
                            self, "Cannot Open Directory",
                            f"Could not automatically open the directory:\n{str(e)}\n\n"
                            f"Please navigate to: {full_package_path}"
                        )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                f"Failed to save configuration:\n{str(e)}")

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
        
        # Get image size from preprocessing config, fallback to default
        preprocessing_cfg = self.gui_cfg.get("preprocessing", {})
        # Try to get target_size from preprocessing, otherwise use default
        size = 224  # Default size
        
        # Look for resizing configuration in preprocessing
        if isinstance(preprocessing_cfg, dict):
            for method_name, method_config in preprocessing_cfg.items():
                if isinstance(method_config, dict) and method_config.get('enabled', False):
                    if 'target_size' in method_config:
                        target_size = method_config['target_size']
                        if isinstance(target_size, list) and len(target_size) >= 2:
                            size = int(target_size[0])
                        elif isinstance(target_size, int):
                            size = target_size
                        break
            
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
        
        # Create a widget to display the preview in the Data Preview tab
        preview_widget = QWidget()
        layout = QGridLayout(preview_widget)
        
        # Original image
        pix = self.np_to_qpixmap(img_rgb)
        w0 = QLabel()
        w0.setPixmap(pix.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        layout.addWidget(QLabel("Original"), 0, 0)
        layout.addWidget(w0, 1, 0)
        
        # Augmented samples
        for i, s in enumerate(samples):
            p = self.np_to_qpixmap(s)
            l = QLabel()
            l.setPixmap(p.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(QLabel(f"Aug {i+1}"), 0, i+1)
            layout.addWidget(l, 1, i+1)
        
        # Set the preview widget in the scroll area
        self.preview_scroll.setWidget(preview_widget)
        
        # Switch to the Data Preview tab
        self.tab_widget.setCurrentIndex(1)  # Index 1 is Data Preview tab

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

    def _connect_tab_handlers(self):
        """Connect button handlers for the new tab interface."""
        # Find buttons and connect them to handlers
        
        # Testing tab buttons
        for btn in self.findChildren(QPushButton):
            if btn.text() == "Test Current Model":
                btn.clicked.connect(self.test_current_model)
            elif btn.text() == "Load Test Dataset":
                btn.clicked.connect(self.load_test_dataset)
            elif btn.text() == "Run Evaluation":
                btn.clicked.connect(self.run_evaluation)
            elif btn.text() == "Export SavedModel":
                btn.clicked.connect(self.export_savedmodel)
            elif btn.text() == "Export TFLite":
                btn.clicked.connect(self.export_tflite)
            elif btn.text() == "Export ONNX":
                btn.clicked.connect(self.export_onnx)
            elif btn.text() == "Deploy Locally":
                btn.clicked.connect(self.deploy_locally)
            elif btn.text() == "Deploy to Cloud":
                btn.clicked.connect(self.deploy_cloud)
            elif btn.text() == "Create Docker Container":
                btn.clicked.connect(self.create_container)
            elif btn.text() == "Load Deployed Model":
                btn.clicked.connect(self.load_deployed_model)
            elif btn.text() == "Select Image":
                btn.clicked.connect(self.select_image_for_prediction)
            elif btn.text() == "Run Prediction":
                btn.clicked.connect(self.run_prediction)
            elif btn.text() == "Batch Prediction":
                btn.clicked.connect(self.batch_prediction)

    # Testing tab methods
    def test_current_model(self):
        """Test the current trained model."""
        self.test_results.appendPlainText("Testing current model...")
        # Implement model testing logic here
        self.append_log("Model testing initiated")

    def load_test_dataset(self):
        """Load a test dataset for evaluation."""
        path = QFileDialog.getExistingDirectory(self, "Select Test Dataset Directory")
        if path:
            self.test_results.appendPlainText(f"Test dataset loaded: {path}")
            self.append_log(f"Test dataset loaded: {path}")

    def run_evaluation(self):
        """Run model evaluation on test dataset."""
        self.test_results.appendPlainText("Running evaluation...")
        # Implement evaluation logic here
        self.append_log("Model evaluation started")

    # Deploy tab methods
    def export_savedmodel(self):
        """Export model as SavedModel format."""
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if path:
            self.deploy_log.appendPlainText(f"Exporting SavedModel to: {path}")
            self.append_log(f"SavedModel export initiated: {path}")

    def export_tflite(self):
        """Export model as TensorFlow Lite format."""
        path, _ = QFileDialog.getSaveFileName(self, "Save TFLite Model", filter="*.tflite")
        if path:
            self.deploy_log.appendPlainText(f"Exporting TFLite model to: {path}")
            self.append_log(f"TFLite export initiated: {path}")

    def export_onnx(self):
        """Export model as ONNX format."""
        path, _ = QFileDialog.getSaveFileName(self, "Save ONNX Model", filter="*.onnx")
        if path:
            self.deploy_log.appendPlainText(f"Exporting ONNX model to: {path}")
            self.append_log(f"ONNX export initiated: {path}")

    def deploy_locally(self):
        """Deploy model locally."""
        self.deploy_log.appendPlainText("Deploying model locally...")
        self.append_log("Local deployment initiated")

    def deploy_cloud(self):
        """Deploy model to cloud."""
        self.deploy_log.appendPlainText("Deploying model to cloud...")
        self.append_log("Cloud deployment initiated")

    def create_container(self):
        """Create Docker container for model deployment."""
        self.deploy_log.appendPlainText("Creating Docker container...")
        self.append_log("Docker container creation initiated")

    # Prediction tab methods
    def load_deployed_model(self):
        """Load a deployed model for prediction."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", filter="SavedModel (*/saved_model.pb);;TFLite (*.tflite);;ONNX (*.onnx)")
        if path:
            self.model_path_label.setText(f"Loaded: {os.path.basename(path)}")
            self.pred_results.appendPlainText(f"Model loaded: {path}")
            self.append_log(f"Model loaded for prediction: {path}")

    def select_image_for_prediction(self):
        """Select an image for prediction."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if path:
            # Load and display the image
            pixmap = QPixmap(path)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.pred_image_label.setPixmap(scaled_pixmap)
            self.pred_results.appendPlainText(f"Image selected: {os.path.basename(path)}")
            self.selected_image_path = path

    def run_prediction(self):
        """Run prediction on selected image."""
        if hasattr(self, 'selected_image_path'):
            self.pred_results.appendPlainText("Running prediction...")
            # Implement prediction logic here
            self.append_log("Prediction initiated")
        else:
            QMessageBox.warning(self, "No Image", "Please select an image first.")

    def batch_prediction(self):
        """Run batch prediction on multiple images."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory with Images")
        if path:
            self.pred_results.appendPlainText(f"Running batch prediction on: {path}")
            self.append_log(f"Batch prediction initiated: {path}")

    def create_comprehensive_config(self):
        """Create a comprehensive configuration structure with Basic and Advanced sections."""
        
        # Basic Configuration - Most commonly used parameters
        basic_config = {
            'data': {
                'train_dir': '',
                'val_dir': '',
                'data_loader': {
                    'type': 'data_loader_group',
                    'name': 'data_loader'
                },
                'preprocessing': {
                    'type': 'preprocessing_group', 
                    'name': 'preprocessing'
                }
            },
            'model': {
                'backbone_type': 'resnet',
                'model_id': 50,
                'dropout_rate': 0.0,
                'activation': 'relu',
                'optimizer': {
                    'type': 'optimizer_group',
                    'name': 'optimizer'
                },
                'loss_functions': {
                    'type': 'loss_functions_group',
                    'name': 'loss_functions'
                },
                'metrics': {
                    'type': 'metrics_group',
                    'name': 'metrics'
                }
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
            'callbacks': {
                'type': 'callbacks_group',
                'name': 'callbacks'
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
            
            # Optimizer tooltips
            'optimizer': 'Configure optimizers for training the neural network',
            
            # Data loader tooltips
            'data_loader': 'Configure custom data loaders for training and validation data',
            
            'adam': 'Adam optimizer - adaptive moment estimation with momentum',
            'sgd': 'Stochastic Gradient Descent optimizer',
            'rmsprop': 'RMSprop optimizer - root mean square propagation',
            'adagrad': 'Adagrad optimizer - adaptive gradient algorithm',
            'adamw': 'AdamW optimizer - Adam with decoupled weight decay',
            'adadelta': 'Adadelta optimizer - adaptive delta algorithm',
            'adamax': 'Adamax optimizer - Adam with infinity norm',
            'nadam': 'Nadam optimizer - Nesterov-accelerated Adam',
            'ftrl': 'FTRL optimizer - Follow The Regularized Leader',
            'custom': 'Custom optimizer loaded from Python file',
            'learning_rate': 'Step size for parameter updates during optimization',
            'beta_1': 'Exponential decay rate for first moment estimates (momentum)',
            'beta_2': 'Exponential decay rate for second moment estimates (variance)',
            'epsilon': 'Small constant to prevent division by zero',
            'amsgrad': 'Whether to use AMSGrad variant of Adam optimizer',
            'momentum': 'Momentum factor for SGD and other momentum-based optimizers',
            'nesterov': 'Whether to use Nesterov momentum in SGD',
            'rho': 'Smoothing constant for RMSprop and Adadelta optimizers',
            'weight_decay': 'Weight decay coefficient for L2 regularization',
            'initial_accumulator_value': 'Starting value for gradient accumulators',
            'learning_rate_power': 'Power for learning rate decay in FTRL',
            'l1_regularization_strength': 'Strength of L1 regularization',
            'l2_regularization_strength': 'Strength of L2 regularization',
            
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
            'model': 'Neural network architecture, optimizer, loss functions, and metrics settings',
            'training': 'Training process configuration including epochs and learning rate',
            'runtime': 'Runtime and system configuration',
            'model_advanced': 'Advanced model architecture parameters',
            'data_advanced': 'Advanced data pipeline configuration',
            'augmentation': 'Data augmentation and preprocessing settings',
            'preprocessing': 'Data preprocessing methods including resizing and normalization',
            'data_loader': 'Custom data loader configuration for training and validation data',
            'optimizer': 'Optimizer configuration for training neural networks',
            'loss_functions': 'Loss function selection and configuration for model training',
            'metrics': 'Training and validation metrics configuration',
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
            
            # Check if this is a special optimizer group type
            if data.get('type') == 'optimizer_group':
                return {
                    'name': data.get('name', name),
                    'type': 'optimizer_group',
                    'tip': self.get_parameter_tooltip('optimizer')
                }
            
            # Check if this is a special data loader group type
            if data.get('type') == 'data_loader_group':
                return {
                    'name': data.get('name', name),
                    'type': 'data_loader_group',
                    'tip': self.get_parameter_tooltip('data_loader')
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
