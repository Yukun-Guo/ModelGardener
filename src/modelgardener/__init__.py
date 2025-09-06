"""
ModelGardener - CLI-only Deep Learning Training Tool

A command-line interface for deep learning model training with TensorFlow/Keras.
Provides modular components for data loading, model building, and training orchestration.
"""

__version__ = "2.0.0"
__author__ = "ModelGardener Team"
__description__ = "A command-line interface for deep learning model training with TensorFlow/Keras"

# Package information
PACKAGE_NAME = "modelgardener"

# Main components
from .enhanced_trainer import EnhancedTrainer
from .enhanced_model_builder import EnhancedModelBuilder
from .scalable_dataset_loader import ScalableDatasetLoader
from .training_components_builder import TrainingComponentsBuilder
from .runtime_configurator import RuntimeConfigurator
from .config_manager import ConfigManager

__all__ = [
    "EnhancedTrainer",
    "EnhancedModelBuilder", 
    "ScalableDatasetLoader",
    "TrainingComponentsBuilder",
    "RuntimeConfigurator",
    "ConfigManager",
]
CLI_NAME = "modelgardener"
