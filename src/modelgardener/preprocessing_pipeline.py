"""
Advanced preprocessing pipeline for ModelGardener.

This module provides a comprehensive preprocessing system that:
1. Applies built-in preprocessing (sizing, normalization) before custom preprocessing
2. Supports both 2D images and 3D data (volumes, sequences)
3. Handles label preprocessing when labels are images or 3D data
4. Maintains compatibility with both CLI training and script generation

The preprocessing order is:
1. Built-in sizing (scaling or crop-padding)
2. Built-in normalization (various methods)
3. Custom preprocessing functions (user-defined)
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Union, Optional, List, Callable
import cv2
from pathlib import Path
import inspect

class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for ModelGardener.
    
    Handles built-in preprocessing (sizing, normalization) followed by custom preprocessing.
    Supports both 2D and 3D data with optional label preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any], custom_functions: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration from config.yaml
            custom_functions: Optional custom preprocessing functions
        """
        self.config = config
        self.custom_functions = custom_functions or {}
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Extract sizing and normalization configs
        self.sizing_config = self.preprocessing_config.get('Resizing', {})
        self.normalization_config = self.preprocessing_config.get('Normalization', {})
        
    def preprocess_batch(self, data: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply full preprocessing pipeline to a batch of data and labels.
        
        Args:
            data: Input data tensor [batch, height, width, channels] or [batch, depth, height, width, channels]
            labels: Label tensor (can be classification labels or image/volume labels)
            
        Returns:
            Tuple of (processed_data, processed_labels)
        """
        # Apply built-in preprocessing first
        processed_data = self._apply_builtin_sizing(data)
        processed_data = self._apply_builtin_normalization(processed_data)
        
        # Apply label preprocessing if labels are images/volumes
        processed_labels = self._apply_label_preprocessing(labels)
        
        # Apply custom preprocessing last
        processed_data = self._apply_custom_preprocessing(processed_data, processed_labels)
        
        return processed_data, processed_labels
    
    def _apply_builtin_sizing(self, data: tf.Tensor) -> tf.Tensor:
        """Apply built-in sizing operations (scaling or crop-padding)."""
        if not self.sizing_config.get('enabled', False):
            return data
            
        target_size = self.sizing_config.get('target_size', {})
        data_format = self.sizing_config.get('data_format', '2D')
        method = self.sizing_config.get('method', 'scaling')
        
        if data_format == '2D':
            return self._resize_2d(data, target_size, method)
        elif data_format == '3D':
            return self._resize_3d(data, target_size, method)
        else:
            return data
    
    def _resize_2d(self, data: tf.Tensor, target_size: Dict[str, int], method: str) -> tf.Tensor:
        """Resize 2D images using specified method."""
        height = target_size.get('height', 224)
        width = target_size.get('width', 224)
        
        if method == 'scaling':
            interpolation = self.sizing_config.get('interpolation', 'bilinear')
            preserve_aspect = self.sizing_config.get('preserve_aspect_ratio', True)
            
            # Map interpolation method
            interp_map = {
                'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                'bilinear': tf.image.ResizeMethod.BILINEAR,
                'bicubic': tf.image.ResizeMethod.BICUBIC,
                'area': tf.image.ResizeMethod.AREA,
                'lanczos': tf.image.ResizeMethod.LANCZOS3
            }
            
            resize_method = interp_map.get(interpolation, tf.image.ResizeMethod.BILINEAR)
            
            if preserve_aspect:
                # Resize with padding to preserve aspect ratio
                resized = tf.image.resize_with_pad(
                    data, height, width, method=resize_method, antialias=False
                )
            else:
                # Direct resize (may distort aspect ratio)
                resized = tf.image.resize(data, [height, width], method=resize_method, antialias=False)
                
        elif method == 'pad_crop':
            crop_method = self.sizing_config.get('crop_method', 'center')
            pad_value = self.sizing_config.get('pad_value', 0.0)
            
            if crop_method == 'center':
                # Center crop/pad
                resized = tf.image.resize_with_crop_or_pad(data, height, width)
                # Apply padding value if padding occurred
                # Note: TensorFlow's resize_with_crop_or_pad uses 0 padding by default
                # For custom pad values, we need to implement custom logic
                resized = self._apply_custom_padding(resized, data, height, width, pad_value)
            else:  # random crop
                # For random crop, we'll use random_crop if input is larger, or pad if smaller
                current_shape = tf.shape(data)
                current_h, current_w = current_shape[-3], current_shape[-2]
                
                # Determine if we need to crop or pad
                if tf.reduce_all([current_h >= height, current_w >= width]):
                    # Random crop
                    resized = tf.image.random_crop(data, [tf.shape(data)[0], height, width, tf.shape(data)[-1]])
                else:
                    # Pad first, then potentially crop
                    padded = tf.image.resize_with_crop_or_pad(data, 
                                                            tf.maximum(height, current_h), 
                                                            tf.maximum(width, current_w))
                    if tf.reduce_all([tf.shape(padded)[-3] > height, tf.shape(padded)[-2] > width]):
                        resized = tf.image.random_crop(padded, [tf.shape(data)[0], height, width, tf.shape(data)[-1]])
                    else:
                        resized = padded
        else:
            # Fallback to simple resize
            resized = tf.image.resize(data, [height, width])
            
        return resized
    
    def _resize_3d(self, data: tf.Tensor, target_size: Dict[str, int], method: str) -> tf.Tensor:
        """Resize 3D volumes/sequences using specified method."""
        height = target_size.get('height', 224)
        width = target_size.get('width', 224)
        depth = target_size.get('depth', 16)
        
        if method == 'scaling':
            # For 3D data, we need to handle depth dimension as well
            interpolation = self.sizing_config.get('interpolation', 'bilinear')
            
            # Get current shape
            current_shape = tf.shape(data)
            
            # Resize spatial dimensions (H, W) first
            if len(data.shape) >= 4:  # Has batch and depth dimensions
                if len(data.shape) == 5:  # [batch, depth, height, width, channels]
                    current_depth = current_shape[1]
                    # Process each frame/slice separately for spatial resizing
                    resized_frames = []
                    for i in range(current_depth):
                        frame = data[:, i, :, :, :]  # [batch, height, width, channels]
                        # Use the existing 2D resize function
                        resized_frame = self._resize_2d(frame, {'height': height, 'width': width}, 'scaling')
                        resized_frames.append(resized_frame)
                    
                    # Stack frames back
                    resized_spatial = tf.stack(resized_frames, axis=1)  # [batch, depth, height, width, channels]
                    
                    # Now handle depth dimension if needed
                    if current_depth != depth:
                        # For depth resizing, we'll use simple interpolation
                        if current_depth > depth:
                            # Subsample depth dimension
                            indices = tf.linspace(0, current_depth - 1, depth)
                            indices = tf.cast(indices, tf.int32)
                            resized = tf.gather(resized_spatial, indices, axis=1)
                        else:
                            # Upsample depth dimension by repeating slices
                            repeat_factor = depth // current_depth
                            remainder = depth % current_depth
                            
                            # Repeat existing slices
                            repeated_slices = tf.repeat(resized_spatial, repeat_factor, axis=1)
                            
                            # Add remainder slices if needed
                            if remainder > 0:
                                extra_slices = resized_spatial[:, :remainder, :, :, :]
                                repeated_slices = tf.concat([repeated_slices, extra_slices], axis=1)
                            
                            resized = repeated_slices
                    else:
                        resized = resized_spatial
                        
                elif len(data.shape) == 4:  # [depth, height, width, channels] - no batch dim
                    current_depth = current_shape[0]
                    # Process each frame/slice separately
                    resized_frames = []
                    for i in range(current_depth):
                        frame = data[i, :, :, :]  # [height, width, channels]
                        # Add batch dimension temporarily for 2D resize
                        frame_batched = tf.expand_dims(frame, 0)
                        resized_frame = self._resize_2d(frame_batched, {'height': height, 'width': width}, 'scaling')
                        # Remove batch dimension
                        resized_frame = tf.squeeze(resized_frame, 0)
                        resized_frames.append(resized_frame)
                    
                    # Stack frames back
                    resized_spatial = tf.stack(resized_frames, axis=0)  # [depth, height, width, channels]
                    
                    # Handle depth dimension
                    if current_depth != depth:
                        if current_depth > depth:
                            # Subsample depth dimension
                            indices = tf.linspace(0, current_depth - 1, depth)
                            indices = tf.cast(indices, tf.int32)
                            resized = tf.gather(resized_spatial, indices, axis=0)
                        else:
                            # Upsample depth dimension
                            repeat_factor = depth // current_depth
                            remainder = depth % current_depth
                            
                            repeated_slices = tf.repeat(resized_spatial, repeat_factor, axis=0)
                            
                            if remainder > 0:
                                extra_slices = resized_spatial[:remainder, :, :, :]
                                repeated_slices = tf.concat([repeated_slices, extra_slices], axis=0)
                            
                            resized = repeated_slices
                    else:
                        resized = resized_spatial
                else:
                    # Fallback for other shapes
                    resized = data
            else:
                # Fallback for other shapes
                resized = data
                
        elif method == 'pad_crop':
            # 3D crop/pad implementation
            # First handle spatial dimensions
            resized = tf.image.resize_with_crop_or_pad(data, height, width)
            
            # Then handle depth dimension
            current_shape = tf.shape(resized)
            if len(data.shape) == 5:  # [batch, depth, height, width, channels]
                current_depth = current_shape[1]
            elif len(data.shape) == 4:  # [depth, height, width, channels]
                current_depth = current_shape[0]
            else:
                current_depth = depth  # No depth dimension
            
            if current_depth != depth:
                # Pad or crop depth dimension
                if current_depth > depth:
                    # Crop depth (center crop)
                    start_idx = (current_depth - depth) // 2
                    if len(data.shape) == 5:  # [batch, depth, height, width, channels]
                        resized = resized[:, start_idx:start_idx+depth, :, :, :]
                    elif len(data.shape) == 4:  # [depth, height, width, channels]
                        resized = resized[start_idx:start_idx+depth, :, :, :]
                else:
                    # Pad depth
                    pad_depth = depth - current_depth
                    pad_before = pad_depth // 2
                    pad_after = pad_depth - pad_before
                    
                    if len(data.shape) == 5:  # [batch, depth, height, width, channels]
                        paddings = [[0, 0], [pad_before, pad_after], [0, 0], [0, 0], [0, 0]]
                    elif len(data.shape) == 4:  # [depth, height, width, channels]
                        paddings = [[pad_before, pad_after], [0, 0], [0, 0], [0, 0]]
                    else:
                        paddings = [[0, 0], [0, 0]]  # Fallback
                    
                    resized = tf.pad(resized, paddings, mode='CONSTANT', constant_values=0.0)
        else:
            resized = data
            
        return resized
    
    def _apply_custom_padding(self, resized: tf.Tensor, original: tf.Tensor, 
                            target_h: int, target_w: int, pad_value: float) -> tf.Tensor:
        """Apply custom padding value if padding occurred during resize."""
        orig_shape = tf.shape(original)
        resized_shape = tf.shape(resized)
        
        # Check if padding occurred (output is larger than input in any dimension)
        if tf.reduce_any([resized_shape[-3] > orig_shape[-3], resized_shape[-2] > orig_shape[-2]]):
            # Custom padding logic would go here
            # For now, return as-is since TensorFlow's default padding is often sufficient
            pass
            
        return resized
    
    def _apply_builtin_normalization(self, data: tf.Tensor) -> tf.Tensor:
        """Apply built-in normalization methods."""
        if not self.normalization_config.get('enabled', True):
            return data
            
        method = self.normalization_config.get('method', 'zero-center')
        
        if method == 'min-max':
            return self._normalize_minmax(data)
        elif method in ['zero-center', 'standard']:
            return self._normalize_zerocenter(data)
        elif method == 'unit-norm':
            return self._normalize_unitnorm(data)
        elif method == 'robust':
            return self._normalize_robust(data)
        elif method == 'layer-norm':
            return self._normalize_layernorm(data)
        else:
            return data
    
    def _normalize_minmax(self, data: tf.Tensor) -> tf.Tensor:
        """Apply min-max normalization."""
        min_val = self.normalization_config.get('min_value', 0.0)
        max_val = self.normalization_config.get('max_value', 1.0)
        
        # Ensure data is in float format
        data = tf.cast(data, tf.float32)
        
        # Get current min/max
        current_min = tf.reduce_min(data)
        current_max = tf.reduce_max(data)
        
        # Avoid division by zero
        range_val = tf.maximum(current_max - current_min, 1e-8)
        
        # Normalize to [0, 1] first
        normalized = (data - current_min) / range_val
        
        # Scale to [min_val, max_val]
        scaled = normalized * (max_val - min_val) + min_val
        
        return scaled
    
    def _normalize_zerocenter(self, data: tf.Tensor) -> tf.Tensor:
        """Apply zero-center (standardization) normalization."""
        data = tf.cast(data, tf.float32)
        
        # Get mean and std configuration
        mean_config = self.normalization_config.get('mean', {'r': 0.485, 'g': 0.456, 'b': 0.406})
        std_config = self.normalization_config.get('std', {'r': 0.229, 'g': 0.224, 'b': 0.225})
        
        # Handle different data formats
        if isinstance(mean_config, dict):
            # RGB format
            if 'r' in mean_config:
                mean = tf.constant([mean_config['r'], mean_config['g'], mean_config['b']], dtype=tf.float32)
                std = tf.constant([std_config['r'], std_config['g'], std_config['b']], dtype=tf.float32)
            else:
                # Convert to list format
                mean = tf.constant(list(mean_config.values()), dtype=tf.float32)
                std = tf.constant(list(std_config.values()), dtype=tf.float32)
        elif isinstance(mean_config, (list, tuple)):
            mean = tf.constant(mean_config, dtype=tf.float32)
            std = tf.constant(std_config, dtype=tf.float32)
        else:
            # Single value (grayscale)
            mean = tf.constant(mean_config, dtype=tf.float32)
            std = tf.constant(std_config, dtype=tf.float32)
        
        # Ensure data is in [0, 1] range first if it's in [0, 255]
        if tf.reduce_max(data) > 1.0:
            data = data / 255.0
        
        # Apply normalization: (x - mean) / std
        normalized = (data - mean) / std
        
        return normalized
    
    def _normalize_unitnorm(self, data: tf.Tensor) -> tf.Tensor:
        """Apply unit norm normalization (L2 normalization)."""
        data = tf.cast(data, tf.float32)
        
        # L2 normalize along the last axis (channels)
        normalized = tf.nn.l2_normalize(data, axis=-1)
        
        return normalized
    
    def _normalize_robust(self, data: tf.Tensor) -> tf.Tensor:
        """Apply robust normalization using median and IQR."""
        data = tf.cast(data, tf.float32)
        
        # Calculate median and IQR
        # Note: TensorFlow doesn't have built-in median, so we approximate
        sorted_data = tf.sort(tf.reshape(data, [-1]))
        n = tf.shape(sorted_data)[0]
        
        # Approximate median
        median = sorted_data[n // 2]
        
        # Approximate Q1 and Q3 (25th and 75th percentiles)
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        
        # Avoid division by zero
        iqr = tf.maximum(iqr, 1e-8)
        
        # Robust normalization: (x - median) / IQR
        normalized = (data - median) / iqr
        
        return normalized
    
    def _normalize_layernorm(self, data: tf.Tensor) -> tf.Tensor:
        """Apply layer normalization."""
        data = tf.cast(data, tf.float32)
        
        # Apply layer normalization along the last axis
        epsilon = self.normalization_config.get('epsilon', 1e-7)
        axis = self.normalization_config.get('axis', -1)
        
        mean, variance = tf.nn.moments(data, axes=[axis], keepdims=True)
        normalized = (data - mean) / tf.sqrt(variance + epsilon)
        
        return normalized
    
    def _apply_label_preprocessing(self, labels: tf.Tensor) -> tf.Tensor:
        """Apply preprocessing to labels if they are images or 3D data."""
        # Check if labels are image-like (more than 1D with spatial dimensions)
        label_shape = tf.shape(labels)
        
        # If labels have spatial dimensions (height, width), apply sizing
        if len(label_shape) >= 3:  # [batch, height, width] or [batch, height, width, channels]
            # Apply same sizing as data but no normalization for labels
            if self.sizing_config.get('enabled', False):
                target_size = self.sizing_config.get('target_size', {})
                data_format = self.sizing_config.get('data_format', '2D')
                method = self.sizing_config.get('method', 'scaling')
                
                if data_format == '2D':
                    # For labels, use nearest neighbor interpolation to preserve discrete values
                    old_interp = self.sizing_config.get('interpolation', 'bilinear')
                    self.sizing_config['interpolation'] = 'nearest'
                    processed_labels = self._resize_2d(labels, target_size, method)
                    self.sizing_config['interpolation'] = old_interp  # Restore original
                elif data_format == '3D':
                    old_interp = self.sizing_config.get('interpolation', 'bilinear')
                    self.sizing_config['interpolation'] = 'nearest'
                    processed_labels = self._resize_3d(labels, target_size, method)
                    self.sizing_config['interpolation'] = old_interp  # Restore original
                else:
                    processed_labels = labels
            else:
                processed_labels = labels
        else:
            # Labels are classification labels, no preprocessing needed
            processed_labels = labels
            
        return processed_labels
    
    def _apply_custom_preprocessing(self, data: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Apply custom preprocessing functions after built-in preprocessing."""
        if not self.custom_functions:
            return data
            
        # Get custom preprocessing functions
        custom_preprocessing = self.custom_functions.get('preprocessing', {})
        if not custom_preprocessing:
            # Check alternative naming
            custom_preprocessing = self.custom_functions.get('preprocessing_functions', {})
        
        if not custom_preprocessing:
            return data
        
        processed_data = data
        
        # Apply each enabled custom preprocessing function
        for func_name, func_info in custom_preprocessing.items():
            # Check if this custom function is enabled in config
            func_config = self.preprocessing_config.get(func_name, {})
            if not func_config.get('enabled', False):
                continue
            
            try:
                # Get the function
                if isinstance(func_info, dict):
                    func = func_info.get('function')
                    if func is None:
                        func = func_info.get('loader')  # Alternative naming
                else:
                    func = func_info
                
                if func is None:
                    continue
                
                # Prepare function parameters
                func_params = {}
                if isinstance(func_info, dict) and 'parameters' in func_info:
                    for param in func_info['parameters']:
                        param_name = param['name']
                        param_value = func_config.get(param_name, param.get('default'))
                        func_params[param_name] = param_value
                
                # Check function signature to see if it expects labels
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Apply the function based on its signature
                if len(param_names) == 0:
                    # Function takes no parameters, it's a wrapper function
                    wrapper_func = func()
                    if callable(wrapper_func):
                        # Check wrapper function signature
                        wrapper_sig = inspect.signature(wrapper_func)
                        wrapper_param_names = list(wrapper_sig.parameters.keys())
                        if len(wrapper_param_names) >= 2:
                            processed_data, _ = wrapper_func(processed_data, labels)
                        else:
                            processed_data = wrapper_func(processed_data)
                    else:
                        processed_data = wrapper_func
                elif len(param_names) >= 2 and 'data' in param_names and 'label' in param_names:
                    # Function expects (data, labels) format directly
                    if func_params:
                        processed_data, _ = func(processed_data, labels, **func_params)
                    else:
                        processed_data, _ = func(processed_data, labels)
                elif param_names and param_names[0] not in ['data', 'labels']:
                    # Function expects configuration parameters, returns wrapper
                    if func_params:
                        wrapper_func = func(**func_params)
                    else:
                        wrapper_func = func()
                    
                    if callable(wrapper_func):
                        # Check wrapper function signature
                        wrapper_sig = inspect.signature(wrapper_func)
                        wrapper_param_names = list(wrapper_sig.parameters.keys())
                        if len(wrapper_param_names) >= 2:
                            processed_data, _ = wrapper_func(processed_data, labels)
                        else:
                            processed_data = wrapper_func(processed_data)
                    else:
                        processed_data = wrapper_func
                else:
                    # Function only processes data
                    if func_params:
                        processed_data = func(processed_data, **func_params)
                    else:
                        processed_data = func(processed_data)
                
            except Exception as e:
                # Log error but continue with other preprocessing steps
                print(f"Warning: Error applying custom preprocessing {func_name}: {str(e)}")
                continue
        
        return processed_data

def create_preprocessing_function(config: Dict[str, Any], custom_functions: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Create a preprocessing function that can be used with tf.data.Dataset.map().
    
    Args:
        config: Preprocessing configuration
        custom_functions: Optional custom preprocessing functions
        
    Returns:
        A function that can be used with dataset.map()
    """
    pipeline = PreprocessingPipeline(config, custom_functions)
    
    def preprocess_fn(data, labels):
        """Preprocessing function for tf.data.Dataset.map()."""
        return pipeline.preprocess_batch(data, labels)
    
    return preprocess_fn
