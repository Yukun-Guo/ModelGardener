"""
Runtime Configuration Module for ModelGardener

This module handles GPU detection, distribution strategy setup, and runtime optimizations.
"""

import os
import tensorflow as tf
from typing import Dict, Any, Tuple
from bridge_callback import BRIDGE


class RuntimeConfigurator:
    """Handles runtime configuration including GPU setup and distribution strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runtime_config = config.get('runtime', {})
    
    def setup_compute_strategy(self) -> Tuple[tf.distribute.Strategy, int]:
        """
        Setup compute strategy based on available hardware and configuration.
        
        Returns:
            Tuple[tf.distribute.Strategy, int]: Strategy and number of GPUs
        """
        try:
            # Configure GPU settings first
            self.configure_gpu_settings()
            
            # Get GPU configuration
            configured_gpus = self.runtime_config.get('num_gpus', 0)
            available_gpus = len(tf.config.list_physical_devices('GPU'))
            
            # Determine actual number of GPUs to use
            if configured_gpus == 0:  # Auto-detect
                num_gpus = available_gpus
            else:
                num_gpus = min(configured_gpus, available_gpus)
            
            BRIDGE.log(f"Available GPUs: {available_gpus}, Using: {num_gpus}")
            
            # Setup distribution strategy
            if num_gpus <= 1:
                strategy = tf.distribute.get_strategy()  # Default strategy
                strategy_name = "Default (CPU/Single GPU)"
            else:
                # Check for specific distribution strategy
                dist_strategy = self.runtime_config.get('distribution_strategy', 'mirrored')
                
                if dist_strategy == 'mirrored':
                    strategy = tf.distribute.MirroredStrategy()
                    strategy_name = "MirroredStrategy"
                elif dist_strategy == 'multi_worker_mirrored':
                    strategy = tf.distribute.MultiWorkerMirroredStrategy()
                    strategy_name = "MultiWorkerMirroredStrategy"
                else:
                    strategy = tf.distribute.MirroredStrategy()  # Default fallback
                    strategy_name = "MirroredStrategy (fallback)"
            
            BRIDGE.log(f"Using distribution strategy: {strategy_name}")
            return strategy, num_gpus
            
        except Exception as e:
            BRIDGE.log(f"Error setting up compute strategy: {str(e)}")
            BRIDGE.log("Falling back to default strategy")
            return tf.distribute.get_strategy(), 0
    
    def configure_gpu_settings(self):
        """Configure GPU memory growth and optimization settings."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                BRIDGE.log(f"Found {len(gpus)} GPU(s)")
                
                for gpu in gpus:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    tf.config.experimental.set_memory_growth(gpu, True)
                    BRIDGE.log(f"Enabled memory growth for {gpu.name}")
                
                # Set additional GPU configurations if specified
                memory_limit = self.runtime_config.get('gpu_memory_limit_mb')
                if memory_limit:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_limit(gpu, memory_limit)
                        BRIDGE.log(f"Set memory limit to {memory_limit}MB for {gpu.name}")
            
            else:
                BRIDGE.log("No GPUs found, using CPU")
                
        except Exception as e:
            BRIDGE.log(f"Error configuring GPU settings: {str(e)}")
    
    def setup_mixed_precision(self):
        """Setup mixed precision training if requested."""
        try:
            if self.runtime_config.get('mixed_precision', False):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                BRIDGE.log("Enabled mixed precision training (float16)")
            else:
                BRIDGE.log("Mixed precision training disabled")
                
        except Exception as e:
            BRIDGE.log(f"Error setting up mixed precision: {str(e)}")
    
    def configure_tensorflow_optimizations(self):
        """Configure TensorFlow optimization settings."""
        try:
            # Set XLA JIT compilation if requested
            if self.runtime_config.get('xla_jit', False):
                tf.config.optimizer.set_jit(True)
                BRIDGE.log("Enabled XLA JIT compilation")
            
            # Set other optimizations
            experimental_opts = self.runtime_config.get('experimental_optimizations', {})
            
            if experimental_opts.get('enable_grappler', True):
                tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': False})
            
            if experimental_opts.get('enable_auto_mixed_precision', False):
                tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
                
        except Exception as e:
            BRIDGE.log(f"Error configuring TensorFlow optimizations: {str(e)}")
    
    def setup_complete_runtime(self) -> Tuple[tf.distribute.Strategy, int]:
        """
        Setup complete runtime configuration.
        
        Returns:
            Tuple[tf.distribute.Strategy, int]: Strategy and number of GPUs
        """
        BRIDGE.log("=== Runtime Configuration ===")
        
        # Setup all runtime components
        self.setup_mixed_precision()
        self.configure_tensorflow_optimizations()
        strategy, num_gpus = self.setup_compute_strategy()
        
        BRIDGE.log("Runtime configuration completed")
        return strategy, num_gpus
