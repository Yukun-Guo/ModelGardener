"""
Custom Function Wrappers for ModelGardener

This module provides standardized wrapper functions for custom functions
to handle parameters consistently across the system.
"""

import inspect
from typing import Dict, Any, Callable


class CustomFunctionWrapper:
    """Base wrapper class for custom functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        """
        Initialize the wrapper.
        
        Args:
            func: The custom function to wrap
            custom_parameters: Dictionary of custom parameters to override defaults
        """
        self.func = func
        self.parameters = self._extract_parameters()
        
        # Override with custom parameters if provided
        if custom_parameters:
            self.parameters.update(custom_parameters)
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameters from the wrapped function."""
        sig = inspect.signature(self.func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            # Skip the first parameter (usually 'data', 'model', etc.)
            if param_name in ['data', 'model', 'self', 'cls']:
                continue
                
            if param.default != inspect.Parameter.empty:
                parameters[param_name] = param.default
                
        return parameters
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped function with parameter handling."""
        return self.func(*args, **kwargs)


class PreprocessingWrapper(CustomFunctionWrapper):
    """Wrapper for preprocessing functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        super().__init__(func, custom_parameters)
        self.function_type = 'preprocessing'
    
    def apply(self, data, config: Dict[str, Any]):
        """
        Apply preprocessing with configuration parameters.
        
        Args:
            data: Input data to preprocess
            config: Configuration parameters from config.yaml
            
        Returns:
            Preprocessed data
        """
        # Extract parameters from config, using function defaults as fallback
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        return self.func(data, **func_params)


class AugmentationWrapper(CustomFunctionWrapper):
    """Wrapper for augmentation functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        super().__init__(func, custom_parameters)
        self.function_type = 'augmentation'
    
    def apply(self, data, config: Dict[str, Any]):
        """
        Apply augmentation with configuration parameters.
        
        Args:
            data: Input data to augment
            config: Configuration parameters from config.yaml
            
        Returns:
            Augmented data
        """
        # Check if augmentation is enabled
        if not config.get('enabled', False):
            return data
        
        # Check probability
        import random
        if random.random() > config.get('probability', 1.0):
            return data
        
        # Extract parameters from config
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        return self.func(data, **func_params)


class CallbackWrapper(CustomFunctionWrapper):
    """Wrapper for callback functions."""
    
    def __init__(self, func_or_class, custom_parameters: Dict[str, Any] = None):
        super().__init__(func_or_class, custom_parameters)
        self.function_type = 'callback'
    
    def create(self, config: Dict[str, Any]):
        """
        Create callback instance with configuration parameters.
        
        Args:
            config: Configuration parameters from config.yaml
            
        Returns:
            Callback instance
        """
        # Extract parameters from config
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        if inspect.isclass(self.func):
            return self.func(**func_params)
        else:
            return self.func(**func_params)


class ModelWrapper(CustomFunctionWrapper):
    """Wrapper for custom model functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        super().__init__(func, custom_parameters)
        self.function_type = 'model'
    
    def create(self, config: Dict[str, Any]):
        """
        Create model with configuration parameters.
        
        Args:
            config: Configuration parameters from config.yaml
            
        Returns:
            Model instance
        """
        # Extract parameters from config
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        return self.func(**func_params)


class DataLoaderWrapper(CustomFunctionWrapper):
    """Wrapper for data loader functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        super().__init__(func, custom_parameters)
        self.function_type = 'data_loader'
    
    def create(self, config: Dict[str, Any]):
        """
        Create data loader with configuration parameters.
        
        Args:
            config: Configuration parameters from config.yaml
            
        Returns:
            Data loader instance
        """
        # Extract parameters from config
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        return self.func(**func_params)


class TrainingLoopWrapper(CustomFunctionWrapper):
    """Wrapper for training loop functions."""
    
    def __init__(self, func: Callable, custom_parameters: Dict[str, Any] = None):
        super().__init__(func, custom_parameters)
        self.function_type = 'training_loop'
    
    def execute(self, model, train_data, val_data, config: Dict[str, Any]):
        """
        Execute training loop with configuration parameters.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data
            config: Configuration parameters from config.yaml
            
        Returns:
            Training history or results
        """
        # Extract parameters from config
        func_params = {}
        for param_name, default_value in self.parameters.items():
            func_params[param_name] = config.get(param_name, default_value)
        
        return self.func(model, train_data, val_data, **func_params)


def wrap_custom_function(func, function_type: str):
    """
    Factory function to create appropriate wrapper for a custom function.
    
    Args:
        func: The custom function to wrap
        function_type: Type of function to determine wrapper class
        
    Returns:
        Appropriate wrapper instance
    """
    wrapper_classes = {
        'preprocessing': PreprocessingWrapper,
        'augmentation': AugmentationWrapper,
        'callback': CallbackWrapper,
        'model': ModelWrapper,
        'data_loader': DataLoaderWrapper,
        'training_loop': TrainingLoopWrapper
    }
    
    wrapper_class = wrapper_classes.get(function_type, CustomFunctionWrapper)
    return wrapper_class(func)
