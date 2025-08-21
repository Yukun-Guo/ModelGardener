"""
Custom Functions Loader - Enhanced functionality for loading custom functions programmatically
"""

import os
import ast
import importlib.util
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QMessageBox


class CustomFunctionsLoader:
    """Helper class to programmatically load custom functions into parameter groups."""
    
    @staticmethod
    def load_custom_data_loader_from_file(data_loader_group, file_path: str, 
                                        function_name: str) -> bool:
        """
        Load a specific custom data loader from a file.
        
        Args:
            data_loader_group: The DataLoaderGroup instance
            file_path: Path to the Python file containing the data loader
            function_name: Name of the function/class to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Parse the Python file to find the specified function/class
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            found_target = False
            target_type = None
            
            # Find the target function or class
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    found_target = True
                    target_type = 'function'
                    break
                elif isinstance(node, ast.ClassDef) and node.name == function_name:
                    found_target = True
                    target_type = 'class'
                    break
            
            if not found_target:
                return False
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_data_loader", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, function_name):
                return False
                
            loader = getattr(module, function_name)
            custom_name = f"Custom_{function_name}"
            
            # Store the function/class
            if not hasattr(data_loader_group, '_custom_data_loaders'):
                data_loader_group._custom_data_loaders = {}
                
            data_loader_group._custom_data_loaders[custom_name] = {
                'loader': loader,
                'type': target_type,
                'file_path': file_path,
                'original_name': function_name
            }
            
            # Extract parameters
            data_loader_group._extract_custom_data_loader_parameters(custom_name, loader, target_type)
            
            # Update data loader options
            data_loader_group._refresh_data_loader_options()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom data loader: {e}")
            return False
    
    @staticmethod
    def load_custom_loss_function_from_file(loss_group, file_path: str, 
                                          function_name: str) -> bool:
        """
        Load a specific custom loss function from a file.
        
        Args:
            loss_group: The LossFunctionsGroup instance
            file_path: Path to the Python file containing the loss function
            function_name: Name of the function/class to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Load and parse the Python file
            custom_functions = loss_group._extract_loss_functions(file_path)
            
            if function_name not in custom_functions:
                return False
            
            func_info = custom_functions[function_name]
            
            # Add custom function to the available loss options
            loss_group._add_custom_loss_option(function_name, func_info)
            
            # Update all loss selection dropdowns
            loss_group._update_all_loss_selections()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom loss function: {e}")
            return False
    
    @staticmethod
    def load_custom_augmentation_from_file(aug_group, file_path: str, 
                                         function_name: str) -> bool:
        """
        Load a specific custom augmentation function from a file.
        
        Args:
            aug_group: The AugmentationGroup instance
            file_path: Path to the Python file containing the augmentation function
            function_name: Name of the function to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Load and parse the Python file
            custom_functions = aug_group._extract_augmentation_functions(file_path)
            
            if function_name not in custom_functions:
                return False
            
            func_info = custom_functions[function_name]
            
            # Add custom function as an augmentation method
            return aug_group._add_custom_function(function_name, func_info)
            
        except Exception as e:
            print(f"Error loading custom augmentation: {e}")
            return False
    
    @staticmethod
    def load_custom_callback_from_file(callback_group, file_path: str, 
                                     function_name: str) -> bool:
        """
        Load a specific custom callback function from a file.
        
        Args:
            callback_group: The CallbacksGroup instance
            file_path: Path to the Python file containing the callback function
            function_name: Name of the function to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Load and parse the Python file
            custom_functions = callback_group._extract_callback_functions(file_path)
            
            if function_name not in custom_functions:
                return False
            
            func_info = custom_functions[function_name]
            
            # Add custom function as a callback method
            return callback_group._add_custom_function(function_name, func_info)
            
        except Exception as e:
            print(f"Error loading custom callback: {e}")
            return False
    
    @staticmethod
    def load_custom_preprocessing_from_file(preprocessing_group, file_path: str, 
                                          function_name: str) -> bool:
        """
        Load a specific custom preprocessing function from a file.
        
        Args:
            preprocessing_group: The PreprocessingGroup instance
            file_path: Path to the Python file containing the preprocessing function
            function_name: Name of the function to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Load and parse the Python file
            custom_functions = preprocessing_group._extract_preprocessing_functions(file_path)
            
            if function_name not in custom_functions:
                return False
            
            func_info = custom_functions[function_name]
            
            # Add custom function as a preprocessing method
            return preprocessing_group._add_custom_function(function_name, func_info)
            
        except Exception as e:
            print(f"Error loading custom preprocessing: {e}")
            return False
    
    @staticmethod
    def load_custom_optimizer_from_file(optimizer_group, file_path: str, 
                                      function_name: str) -> bool:
        """
        Load a specific custom optimizer from a file.
        
        Args:
            optimizer_group: The OptimizerGroup instance
            file_path: Path to the Python file containing the optimizer function
            function_name: Name of the function to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_optimizer", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, function_name):
                return False
                
            func = getattr(module, function_name)
            custom_name = f"Custom_{function_name}"
            
            # Store the function
            if not hasattr(optimizer_group, '_custom_optimizers'):
                optimizer_group._custom_optimizers = {}
            optimizer_group._custom_optimizers[custom_name] = func
            
            # Extract parameters from function signature
            optimizer_group._extract_custom_optimizer_parameters(custom_name, func)
            
            # Update optimizer options
            optimizer_group._refresh_optimizer_options()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom optimizer: {e}")
            return False
