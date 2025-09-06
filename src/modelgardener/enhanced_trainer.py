"""
Refactored Enhanced Trainer - Main orchestrator for the training pipeline

This module provides the main training orchestrator that coordinates all components:
- Runtime configuration (GPU, distribution strategies, optimizations)
- Data pipeline creation with generator patterns and optimizations
- Model building and compilation
- Training execution (standard, cross-validation, custom loops)
"""

import os
import tensorflow as tf
import keras
from typing import Dict, Any, Tuple, Optional

# Import our modular components using relative imports
from .bridge_callback import BRIDGE
from .runtime_configurator import RuntimeConfigurator
from .scalable_dataset_loader import ScalableDatasetLoader
from .enhanced_model_builder import EnhancedModelBuilder
from .training_components_builder import TrainingComponentsBuilder


class EnhancedTrainer:
    """
    Main trainer class that orchestrates the complete training pipeline.
    
    This class provides a unified interface for training that handles:
    - Runtime configuration (GPU, distribution strategies)
    - Data pipeline creation (with generators and optimizations)
    - Model building and compilation
    - Training component setup (callbacks, cross-validation)
    - Training execution (standard, cross-validation, custom loops)
    """
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        """
        Initialize the refactored trainer.
        
        Args:
            config: Complete configuration dictionary
            custom_functions: Dictionary of custom functions (models, data loaders, etc.)
        """
        self.config = config
        self.custom_functions = custom_functions or {}
        
        # Auto-load custom functions if not provided
        print(f"Initial custom_functions: {self.custom_functions}")
        if not self.custom_functions or not any(self.custom_functions.values()):
            print("Auto-loading custom functions...")
            self.custom_functions = self._auto_load_custom_functions()
            print(f"Loaded custom functions: {list(self.custom_functions.keys())}")
        else:
            print(f"Using provided custom functions: {list(self.custom_functions.keys())}")
            # Let's also check if they have the right content
            for key, value in self.custom_functions.items():
                print(f"  {key}: {list(value.keys()) if isinstance(value, dict) else type(value)}")
            
            # Fix the structure if needed
            self.custom_functions = self._fix_custom_functions_structure(self.custom_functions)
        
        # Initialize modular components
        self.runtime_configurator = RuntimeConfigurator(config)
        self.dataset_loader = ScalableDatasetLoader(config, self.custom_functions)
        self.model_builder = EnhancedModelBuilder(config, self.custom_functions)
        self.components_builder = TrainingComponentsBuilder(config, self.custom_functions)
        
        # Training state
        self.strategy = None
        self.num_gpus = 0
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
    
    def train(self) -> bool:
        """
        Main training entry point - unified interface for all training scenarios.
        
        Returns:
            bool: True if training completed successfully, False otherwise
        """
        try:
            print("🚀 Starting Enhanced ModelGardener Training Pipeline")
            print("=" * 60)
            
            # Phase 1: Runtime Setup
            success = self._setup_runtime()
            if not success:
                return False
            
            # Phase 2: Data Pipeline Creation
            success = self._create_data_pipeline()
            if not success:
                return False
            
            # Phase 3: Model Building
            success = self._build_and_compile_model()
            if not success:
                return False
            
            # Phase 4: Training Execution
            success = self._execute_training()
            if not success:
                return False
            
            print("=" * 60)
            print("✅ Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed with error: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
        
        finally:
            self._cleanup_resources()
    
    def _auto_load_custom_functions(self) -> Dict[str, Any]:
        """Auto-load custom functions from custom_modules directory (compatible with package structure)."""
        custom_functions = {'data_loaders': {}, 'models': {}, 'loss_functions': {}, 'metrics': {}, 'callbacks': {}, 'optimizers': {}}
        
        try:
            import importlib.util
            import inspect
            
            # First try to load from custom_modules directory (for generated projects)
            custom_modules_dir = "./custom_modules"
            if os.path.exists(custom_modules_dir):
                print("🔧 Loading custom functions from custom_modules directory...")
                return self._load_from_custom_modules(custom_modules_dir)
            
            # Fallback: try to load from package example_funcs (for development)
            try:
                from modelgardener.example_funcs import example_custom_data_loaders
                print("🔧 Loading custom functions from package example_funcs...")
                return self._load_from_package_examples()
            except ImportError:
                pass
            
            # Final fallback: try relative path to example_funcs
            example_funcs_dir = os.path.join(os.path.dirname(__file__), 'example_funcs')
            if os.path.exists(example_funcs_dir):
                print("🔧 Loading custom functions from example_funcs directory...")
                return self._load_from_example_funcs(example_funcs_dir)
                        
        except Exception as e:
            print(f"Warning: Could not auto-load custom functions: {str(e)}")
            
        return custom_functions
    
    def _load_from_custom_modules(self, custom_modules_dir: str) -> Dict[str, Any]:
        """Load custom functions from custom_modules directory."""
        custom_functions = {'data_loaders': {}, 'models': {}, 'loss_functions': {}, 'metrics': {}, 'callbacks': {}, 'optimizers': {}}
        
        try:
            import importlib.util
            import inspect
            
            # Map custom module files to function types
            custom_files_map = {
                'custom_data_loaders.py': 'data_loaders',
                'custom_models.py': 'models',
                'custom_loss_functions.py': 'loss_functions',
                'custom_metrics.py': 'metrics',
                'custom_callbacks.py': 'callbacks',
                'custom_optimizers.py': 'optimizers',
                'custom_augmentations.py': 'augmentations',
                'custom_preprocessing.py': 'preprocessing'
            }
            
            for custom_file, func_type in custom_files_map.items():
                custom_file_path = os.path.join(custom_modules_dir, custom_file)
                if os.path.exists(custom_file_path):
                    try:
                        spec = importlib.util.spec_from_file_location(f"custom_{func_type}", custom_file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find all functions and classes in the module
                        for name, obj in inspect.getmembers(module):
                            if not name.startswith('_'):  # Skip private functions
                                if inspect.isfunction(obj) or inspect.isclass(obj):
                                    if func_type not in custom_functions:
                                        custom_functions[func_type] = {}
                                    
                                    custom_functions[func_type][name] = {
                                        'loader': obj,
                                        'type': 'function' if inspect.isfunction(obj) else 'class',
                                        'file_path': custom_file_path,
                                        'original_name': name
                                    }
                                    print(f"Auto-loaded {func_type}: {name}")
                    except Exception as e:
                        print(f"Warning: Error loading {custom_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error in _load_from_custom_modules: {str(e)}")
        
        return custom_functions
    
    def _load_from_package_examples(self) -> Dict[str, Any]:
        """Load custom functions from package example_funcs."""
        custom_functions = {'data_loaders': {}, 'models': {}, 'loss_functions': {}, 'metrics': {}, 'callbacks': {}, 'optimizers': {}}
        
        try:
            import inspect
            from modelgardener.example_funcs import example_custom_data_loaders
            
            # Load data loaders
            for name, obj in inspect.getmembers(example_custom_data_loaders):
                if inspect.isfunction(obj) and (name.startswith('load_') or name.startswith('Custom_')):
                    custom_functions['data_loaders'][name] = {
                        'loader': obj,
                        'type': 'function',
                        'file_path': 'modelgardener.example_funcs.example_custom_data_loaders',
                        'original_name': name
                    }
                    print(f"Auto-loaded data loader: {name}")
                elif inspect.isclass(obj) and 'DataLoader' in name:
                    custom_functions['data_loaders'][f"Custom_{name}"] = {
                        'loader': obj,
                        'type': 'class',
                        'file_path': 'modelgardener.example_funcs.example_custom_data_loaders',
                        'original_name': name
                    }
                    print(f"Auto-loaded data loader class: {name}")
        except Exception as e:
            print(f"Warning: Error loading from package examples: {str(e)}")
        
        return custom_functions
    
    def _load_from_example_funcs(self, example_funcs_dir: str) -> Dict[str, Any]:
        """Load custom functions from example_funcs directory."""
        custom_functions = {'data_loaders': {}, 'models': {}, 'loss_functions': {}, 'metrics': {}, 'callbacks': {}, 'optimizers': {}}
        
        try:
            import importlib.util
            import inspect
            
            # Load data loaders from example_funcs directory
            data_loader_file = os.path.join(example_funcs_dir, "example_custom_data_loaders.py")
            if os.path.exists(data_loader_file):
                spec = importlib.util.spec_from_file_location("example_data_loaders", data_loader_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find data loader functions
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and (name.startswith('load_') or name.startswith('Custom_')):
                        custom_functions['data_loaders'][name] = {
                            'loader': obj,
                            'type': 'function',
                            'file_path': data_loader_file,
                            'original_name': name
                        }
                        print(f"Auto-loaded data loader: {name}")
                    elif inspect.isclass(obj) and 'DataLoader' in name:
                        custom_functions['data_loaders'][f"Custom_{name}"] = {
                            'loader': obj,
                            'type': 'class',
                            'file_path': data_loader_file,
                            'original_name': name
                        }
                        print(f"Auto-loaded data loader class: {name}")
        except Exception as e:
            print(f"Warning: Error loading from example_funcs: {str(e)}")
            
        return custom_functions
    
    def _fix_custom_functions_structure(self, custom_functions: Dict[str, Any]) -> Dict[str, Any]:
        """Fix custom functions structure to have actual function objects."""
        fixed_functions = {'data_loaders': {}, 'models': {}, 'loss_functions': {}, 'metrics': {}, 'callbacks': {}, 'optimizers': {}}
        
        try:
            import importlib.util
            
            # Handle both list and dict formats for custom functions
            for category, functions_data in custom_functions.items():
                if category not in fixed_functions:
                    continue
                    
                # Handle list format (from config.yaml metadata)
                if isinstance(functions_data, list):
                    for func_info in functions_data:
                        func_name = func_info.get('name', '')
                        file_path = func_info.get('file_path', '')
                        function_name = func_info.get('function_name', '')
                        func_type = func_info.get('type', 'function')
                        
                        if file_path and function_name and func_name:
                            try:
                                # Load the actual function
                                if os.path.exists(file_path):
                                    spec = importlib.util.spec_from_file_location("custom_module", file_path)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)
                                    
                                    if hasattr(module, function_name):
                                        func = getattr(module, function_name)
                                        fixed_functions[category][func_name] = {
                                            'loader' if category == 'data_loaders' else 'function': func,
                                            'type': func_type,
                                            'file_path': file_path,
                                            'original_name': function_name
                                        }
                                        print(f"Fixed {category} structure: {func_name}")
                            except Exception as e:
                                print(f"Warning: Could not load {category} function {function_name}: {str(e)}")
                                
                # Handle dict format (old format)
                elif isinstance(functions_data, dict):
                    for func_name, func_info in functions_data.items():
                        if 'loader' not in func_info and 'function' not in func_info and 'file_path' in func_info:
                            # Need to load the actual function
                            file_path = func_info['file_path']
                            function_name = func_info.get('function_name', func_info.get('original_name', ''))
                            
                            if os.path.exists(file_path) and function_name:
                                try:
                                    spec = importlib.util.spec_from_file_location("custom_loader", file_path)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)
                                    
                                    if hasattr(module, function_name):
                                        func = getattr(module, function_name)
                                        fixed_functions[category][func_name] = {
                                            'loader' if category == 'data_loaders' else 'function': func,
                                            'type': func_info.get('type', 'function'),
                                            'file_path': file_path,
                                            'original_name': function_name
                                        }
                                        print(f"Fixed {category} structure: {func_name}")
                                except Exception as e:
                                    print(f"Warning: Could not load {category} function {function_name}: {str(e)}")
                        else:
                            # Already has correct structure
                            fixed_functions[category][func_name] = func_info
                    
        except Exception as e:
            print(f"Warning: Could not fix custom functions structure: {str(e)}")
            return custom_functions
            
        return fixed_functions
    
    def _setup_runtime(self) -> bool:
        """Phase 1: Setup runtime configuration."""
        
        try:
            print("📋 Phase 1: Runtime Configuration")
            
            # Setup complete runtime (GPU, distribution, optimizations)
            self.strategy, self.num_gpus = self.runtime_configurator.setup_complete_runtime()
            
            print(f"✅ Runtime setup completed")
            print(f"   • Strategy: {self.strategy.__class__.__name__}")
            print(f"   • GPUs: {self.num_gpus}")
            
            return True
            
        except Exception as e:
            print(f"❌ Runtime setup failed: {str(e)}")
            return False
    
    def _create_data_pipeline(self) -> bool:
        """Phase 2: Create optimized data pipeline."""
        
        try:
            print("📊 Phase 2: Data Pipeline Creation")
            
            # Load training dataset
            self.train_dataset = self.dataset_loader.load_dataset('train')
            print("✅ Training dataset loaded")
            
            # Load validation dataset if available
            if self._has_validation_data():
                self.val_dataset = self.dataset_loader.load_dataset('val')
                print("✅ Validation dataset loaded")
            else:
                self.val_dataset = None
                print("ℹ️  No validation dataset specified")
            
            # Log dataset information
            self._log_dataset_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Data pipeline creation failed: {str(e)}")
            return False
    
    def _build_and_compile_model(self) -> bool:
        """Phase 3: Build and compile model."""
        
        try:
            print("🏗️ Phase 3: Model Building")
            
            with self.strategy.scope():
                # Infer data specifications with fallback to config
                try:
                    input_shape, num_classes = self.dataset_loader.infer_data_specs()
                except Exception as e:
                    print(f"Warning: Could not infer data specs, using config: {str(e)}")
                    # Fallback to model configuration
                    model_params = self.config.get('model', {}).get('model_parameters', {})
                    input_shape_config = model_params.get('input_shape', {})
                    
                    if isinstance(input_shape_config, dict):
                        height = input_shape_config.get('height', 32)
                        width = input_shape_config.get('width', 32) 
                        channels = input_shape_config.get('channels', 3)
                        input_shape = (height, width, channels)
                    else:
                        input_shape = (32, 32, 3)  # Default for CIFAR-10
                        
                    num_classes = model_params.get('num_classes', 10)
                    print(f"Using config input shape: {input_shape}, classes: {num_classes}")
                
                # Build complete model (architecture + compilation)
                self.model = self.model_builder.build_complete_model(input_shape, num_classes)
                
                print("✅ Model built and compiled successfully")
                
                return True
                
        except Exception as e:
            print(f"❌ Model building failed: {str(e)}")
            return False
    
    def _execute_training(self) -> bool:
        """Phase 4: Execute training based on configuration."""
        
        try:
            print("🏃 Phase 4: Training Execution")
            
            # Setup training components
            training_config = self.config.get('training', {})
            epochs = training_config.get('epochs', 100)
            
            # Estimate total steps for progress tracking
            total_steps = self.components_builder.estimate_total_steps(self.train_dataset, epochs)
            
            # Setup callbacks
            callbacks = self.components_builder.setup_training_callbacks(total_steps)
            
            # Determine training strategy
            if self.components_builder.should_use_cross_validation():
                return self._train_with_cross_validation(callbacks)
            elif self.components_builder.should_use_custom_training_loop():
                return self._train_with_custom_loop(callbacks)
            else:
                return self._train_standard(callbacks)
                
        except Exception as e:
            print(f"❌ Training execution failed: {str(e)}")
            return False
    
    def _train_standard(self, callbacks) -> bool:
        """Execute standard training with model.fit()."""
        
        try:
            print("🎯 Executing Standard Training")
            
            training_config = self.config.get('training', {})
            epochs = training_config.get('epochs', 100)
            
            with self.strategy.scope():
                # Run training
                history = self.model.fit(
                    self.train_dataset,
                    validation_data=self.val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save final model
                self._save_final_model()
                
                # Log training summary
                self._log_training_summary(history)
                
                return True
                
        except Exception as e:
            print(f"❌ Standard training failed: {str(e)}")
            return False
    
    def _train_with_cross_validation(self, callbacks) -> bool:
        """Execute training with k-fold cross-validation."""
        
        try:
            print("🔄 Executing Cross-Validation Training")
            
            cv_config = self.components_builder.get_cv_config()
            k_folds = cv_config['k_folds']
            stratified = cv_config['stratified']
            save_fold_models = cv_config['save_fold_models']
            
            # Create CV folds from full dataset
            full_dataset = self.train_dataset
            folds = self.components_builder.create_cv_folds(full_dataset, k_folds, stratified)
            
            fold_results = []
            training_config = self.config.get('training', {})
            epochs = training_config.get('epochs', 100)
            
            for fold_idx, (train_fold, val_fold) in enumerate(folds):
                print(f"📂 Training Fold {fold_idx + 1}/{k_folds}")
                
                with self.strategy.scope():
                    # Reset model for each fold (rebuild from scratch)
                    input_shape, num_classes = self.dataset_loader.infer_data_specs()
                    fold_model = self.model_builder.build_complete_model(input_shape, num_classes)
                    
                    # Train on this fold
                    fold_history = fold_model.fit(
                        train_fold,
                        validation_data=val_fold,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Evaluate fold
                    val_metrics = fold_model.evaluate(val_fold, verbose=0)
                    fold_results.append(val_metrics)
                    
                    print(f"✅ Fold {fold_idx + 1} completed")
                    
                    # Save fold model if requested
                    if save_fold_models:
                        fold_model_path = os.path.join(
                            self.config.get('runtime', {}).get('model_dir', './logs'),
                            f'fold_{fold_idx + 1}_model.keras'
                        )
                        fold_model.save(fold_model_path)
                        print(f"💾 Saved fold model: {fold_model_path}")
            
            # Log cross-validation results
            self.components_builder.log_cv_results(fold_results)
            
            # Train final model on full dataset if requested
            print("🔄 Training final model on full dataset")
            with self.strategy.scope():
                final_history = self.model.fit(
                    full_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
            
            self._save_final_model()
            self._log_training_summary(final_history)
            
            return True
            
        except Exception as e:
            print(f"❌ Cross-validation training failed: {str(e)}")
            return False
    
    def _train_with_custom_loop(self, callbacks) -> bool:
        """Execute training with custom training loop."""
        
        try:
            print("🔧 Executing Custom Training Loop")
            
            custom_loop_info = self.components_builder.get_custom_training_loop_info()
            if not custom_loop_info:
                print("❌ Custom training loop not found, falling back to standard training")
                return self._train_standard(callbacks)
            
            custom_loop_func = custom_loop_info['function']
            training_config = self.config.get('training', {})
            epochs = training_config.get('epochs', 100)
            
            # Prepare arguments for custom training loop
            loop_args = {
                'model': self.model,
                'train_dataset': self.train_dataset,
                'val_dataset': self.val_dataset,
                'epochs': epochs,
                'callbacks': callbacks,
                'config': self.config,
                'strategy': self.strategy
            }
            
            # Execute custom training loop
            with self.strategy.scope():
                if custom_loop_info['type'] == 'function':
                    result = custom_loop_func(**loop_args)
                elif custom_loop_info['type'] == 'class':
                    loop_instance = custom_loop_func(**loop_args)
                    if hasattr(loop_instance, 'run'):
                        result = loop_instance.run()
                    elif hasattr(loop_instance, '__call__'):
                        result = loop_instance()
                    else:
                        raise ValueError("Custom training loop class must have 'run' or '__call__' method")
                else:
                    raise ValueError(f"Unknown custom loop type: {custom_loop_info['type']}")
            
            self._save_final_model()
            print("✅ Custom training loop completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Custom training loop failed: {str(e)}")
            print("🔄 Falling back to standard training")
            return self._train_standard(callbacks)
    
    def _has_validation_data(self) -> bool:
        """Check if validation data is configured."""
        
        data_config = self.config.get('data', {})
        
        # Check for validation directory
        val_dir = data_config.get('val_dir') or data_config.get('val_data')
        if val_dir and os.path.exists(val_dir):
            return True
        
        # Check for validation split in custom loader
        data_loader_config = data_config.get('data_loader', {})
        if 'validation_split' in data_loader_config.get('parameters', {}):
            validation_split = data_loader_config['parameters']['validation_split']
            return validation_split > 0.0
        
        return False
    
    def _log_dataset_info(self):
        """Log information about loaded datasets."""
        
        try:
            # Try to get dataset cardinality for logging
            if self.train_dataset:
                train_cardinality = tf.data.experimental.cardinality(self.train_dataset).numpy()
                if train_cardinality > 0:
                    print(f"   • Training batches: {train_cardinality}")
                else:
                    print(f"   • Training dataset: Unknown size")
            
            if self.val_dataset:
                val_cardinality = tf.data.experimental.cardinality(self.val_dataset).numpy()
                if val_cardinality > 0:
                    print(f"   • Validation batches: {val_cardinality}")
                else:
                    print(f"   • Validation dataset: Unknown size")
                    
        except Exception as e:
            print(f"   • Dataset info: {str(e)}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        
        try:
            model_dir = self.config.get('runtime', {}).get('model_dir', './logs')
            os.makedirs(model_dir, exist_ok=True)
            
            final_model_path = os.path.join(model_dir, 'final_model.keras')
            self.model.save(final_model_path)
            print(f"💾 Final model saved: {final_model_path}")
            
            # # Also save in SavedModel format for deployment
            # savedmodel_path = os.path.join(model_dir, 'savedmodel')
            # self.model.export(savedmodel_path,verbose=0)
            # print(f"💾 SavedModel saved: {savedmodel_path}")
            
        except Exception as e:
            print(f"⚠️  Error saving model: {str(e)}")
    
    def _log_training_summary(self, history):
        """Log training summary and metrics."""
        
        try:
            if hasattr(history, 'history') and history.history:
                print("📊 Training Summary:")
                
                final_epoch = len(history.history.get('loss', []))
                print(f"   • Total epochs: {final_epoch}")
                
                # Log final metrics
                for metric_name, values in history.history.items():
                    if values:
                        final_value = values[-1]
                        print(f"   • Final {metric_name}: {final_value:.4f}")
                
                # Find best validation metrics if available
                if 'val_loss' in history.history:
                    best_val_loss = min(history.history['val_loss'])
                    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
                    print(f"   • Best val_loss: {best_val_loss:.4f} (epoch {best_epoch})")
                
                if 'val_accuracy' in history.history:
                    best_val_acc = max(history.history['val_accuracy'])
                    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
                    print(f"   • Best val_accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
                    
        except Exception as e:
            print(f"⚠️  Error logging training summary: {str(e)}")
    
    def _cleanup_resources(self):
        """Cleanup resources after training."""
        
        try:
            # Clear any cached datasets to free memory
            if hasattr(tf.data.experimental, 'clear_dataset_cache'):
                tf.data.experimental.clear_dataset_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("🧹 Resources cleaned up")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {str(e)}")
    
    # Additional utility methods for compatibility and evaluation
    
    def evaluate(self, dataset: Optional[tf.data.Dataset] = None) -> Optional[Dict[str, float]]:
        """
        Evaluate the trained model.
        
        Args:
            dataset: Dataset to evaluate on (uses validation dataset if None)
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        
        if self.model is None:
            print("❌ No model available for evaluation")
            return None
        
        eval_dataset = dataset or self.val_dataset
        if eval_dataset is None:
            print("❌ No dataset available for evaluation")
            return None
        
        try:
            print("📊 Starting model evaluation...")
            
            # Run evaluation
            results = self.model.evaluate(eval_dataset, verbose=1, return_dict=True)
            
            print("✅ Evaluation completed")
            for metric, value in results.items():
                print(f"   • {metric}: {value:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return None
    
    def predict(self, dataset: tf.data.Dataset, save_predictions: bool = False) -> Optional[tf.Tensor]:
        """
        Generate predictions using the trained model.
        
        Args:
            dataset: Dataset to predict on
            save_predictions: Whether to save predictions to file
            
        Returns:
            tf.Tensor: Model predictions
        """
        
        if self.model is None:
            print("❌ No model available for prediction")
            return None
        
        try:
            print("🔮 Generating predictions...")
            
            predictions = self.model.predict(dataset, verbose=1)
            
            if save_predictions:
                model_dir = self.config.get('runtime', {}).get('model_dir', './logs')
                pred_path = os.path.join(model_dir, 'predictions.npy')
                import numpy as np
                np.save(pred_path, predictions)
                print(f"💾 Predictions saved: {pred_path}")
            
            print("✅ Prediction completed")
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def _load_saved_model(self) -> bool:
        """
        Load a saved model for evaluation/prediction.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_dir = self.config.get('runtime', {}).get('model_dir', './logs')
            
            # Try different model file formats
            model_paths = [
                os.path.join(model_dir, 'final_model.keras'),
                os.path.join(model_dir, 'best_model.keras'),
                os.path.join(model_dir, 'savedmodel'),
                model_dir  # In case model_dir is the model path itself
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"🔄 Loading model from {model_path}")
                    try:
                        if model_path.endswith('.keras'):
                            self.model = keras.models.load_model(model_path)
                        else:
                            # Try loading as SavedModel format
                            self.model = tf.keras.models.load_model(model_path)
                        
                        print(f"✅ Model loaded successfully from {model_path}")
                        return True
                        
                    except Exception as e:
                        print(f"⚠️  Failed to load model from {model_path}: {str(e)}")
                        continue
            
            print(f"❌ No valid model found in {model_dir}")
            return False
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            return False
