"""
Enhanced custom preprocessing functions for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (data, label) and returns (processed_data, processed_label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def preprocessing_name(param1=default1, param2=default2):
    def wrapper(data, label):
        # Apply preprocessing logic here
        processed_data = apply_preprocessing(data, param1, param2)
        processed_label = process_label_if_needed(label)
        return processed_data, processed_label
    return wrapper

NOTE: With the new preprocessing pipeline, built-in preprocessing (sizing, normalization) 
is applied BEFORE custom preprocessing functions. This ensures consistent behavior 
and proper compatibility with 3D data and label preprocessing.
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Union, List, Dict, Tuple, Any
from .utils import (
    TaskType, DataDimension, detect_data_dimension, infer_task_type,
    handle_multi_input, handle_multi_output, apply_2d_operation_to_3d
)

def enhanced_adaptive_histogram_equalization(clip_limit=2.0, tile_grid_size=8, apply_to_3d=True):
    """
    Enhanced adaptive histogram equalization with 2D/3D support and multi-input handling.
    
    Args:
        clip_limit (float): Threshold for contrast limiting (default: 2.0)
        tile_grid_size (int): Size of the grid for adaptive equalization (default: 8)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_clahe_single(data):
        """Apply CLAHE to a single data tensor."""
        data_dim = detect_data_dimension(data)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(data, lambda x: apply_clahe_2d(x, clip_limit, tile_grid_size))
        else:
            return apply_clahe_2d(data, clip_limit, tile_grid_size)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_clahe_single)
        
        # Labels typically don't change for histogram equalization
        return processed_data, label
    
    return wrapper

def apply_clahe_2d(data, clip_limit, tile_grid_size):
    """Apply CLAHE to 2D data using TensorFlow operations."""
    # For TensorFlow implementation, we'll use a simplified approach
    # In practice, you might want to use tf.py_function to call OpenCV
    
    # Convert to numpy for OpenCV operations
    if tf.is_tensor(data):
        # Use tf.py_function to call OpenCV CLAHE
        def clahe_opencv(image_np):
            if len(image_np.shape) == 2:  # Grayscale
                clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                       tileGridSize=(tile_grid_size, tile_grid_size))
                processed = clahe.apply(image_np.astype(np.uint8))
            elif len(image_np.shape) == 3:  # Color image
                if image_np.max() <= 1.0:  # Normalized image
                    image_np = (image_np * 255).astype(np.uint8)
                lab = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                       tileGridSize=(tile_grid_size, tile_grid_size))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                if data.dtype == tf.float32:
                    processed = processed.astype(np.float32) / 255.0
            else:
                processed = image_np
            return processed.astype(np.float32)
        
        processed = tf.py_function(clahe_opencv, [data], tf.float32)
        processed.set_shape(data.shape)
        return processed
    else:
        return data

def z_score_normalization(mean=None, std=None, axis=None, apply_to_3d=True):
    """
    Z-score normalization with support for 2D/3D data and multi-inputs.
    
    Args:
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed standard deviation (if None, computed from data)  
        axis: Axis along which to compute statistics
        apply_to_3d: Whether to apply to 3D data
    """
    def apply_zscore_single(data):
        """Apply z-score normalization to single data tensor."""
        # Compute mean and std if not provided
        data_mean = mean if mean is not None else tf.reduce_mean(data, axis=axis, keepdims=True)
        data_std = std if std is not None else tf.math.reduce_std(data, axis=axis, keepdims=True)
        
        # Avoid division by zero
        data_std = tf.maximum(data_std, 1e-8)
        
        # Apply normalization
        normalized = (data - data_mean) / data_std
        return normalized
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_zscore_single)
        return processed_data, label
    
    return wrapper

def min_max_normalization(min_val=0.0, max_val=1.0, data_min=None, data_max=None, apply_to_3d=True):
    """
    Min-max normalization with support for 2D/3D data and multi-inputs.
    
    Args:
        min_val: Target minimum value
        max_val: Target maximum value
        data_min: Pre-computed data minimum (if None, computed from data)
        data_max: Pre-computed data maximum (if None, computed from data)
        apply_to_3d: Whether to apply to 3D data
    """
    def apply_minmax_single(data):
        """Apply min-max normalization to single data tensor."""
        # Compute min and max if not provided
        current_min = data_min if data_min is not None else tf.reduce_min(data)
        current_max = data_max if data_max is not None else tf.reduce_max(data)
        
        # Avoid division by zero
        range_val = tf.maximum(current_max - current_min, 1e-8)
        
        # Apply normalization
        normalized = (data - current_min) / range_val
        normalized = normalized * (max_val - min_val) + min_val
        
        return normalized
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_minmax_single)
        return processed_data, label
    
    return wrapper

def gaussian_blur(kernel_size=5, sigma=1.0, apply_to_3d=True):
    """
    Gaussian blur preprocessing with support for 2D/3D data and multi-inputs.
    
    Args:
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
        apply_to_3d: Whether to apply to 3D data
    """
    def apply_blur_single(data):
        """Apply Gaussian blur to single data tensor."""
        data_dim = detect_data_dimension(data)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(data, lambda x: apply_gaussian_blur_2d(x, kernel_size, sigma))
        else:
            return apply_gaussian_blur_2d(data, kernel_size, sigma)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_blur_single)
        return processed_data, label
    
    return wrapper

def apply_gaussian_blur_2d(data, kernel_size, sigma):
    """Apply Gaussian blur to 2D data."""
    # Use TensorFlow's Gaussian blur implementation
    # This is a simplified version - for production, use proper Gaussian kernel
    
    channels = tf.shape(data)[-1]
    
    # Create simple averaging kernel as approximation
    kernel_weight = 1.0 / (kernel_size * kernel_size)
    kernel = tf.ones([kernel_size, kernel_size, channels, 1]) * kernel_weight
    
    # Apply blur using depthwise convolution
    if len(data.shape) == 3:  # (H, W, C)
        data_batch = tf.expand_dims(data, 0)
        blurred_batch = tf.nn.depthwise_conv2d(
            data_batch, kernel, strides=[1, 1, 1, 1], padding='SAME'
        )
        return blurred_batch[0]
    else:  # Already batched
        return tf.nn.depthwise_conv2d(
            data, kernel, strides=[1, 1, 1, 1], padding='SAME'
        )

def edge_enhancement(strength=1.0, method='sobel', apply_to_3d=True):
    """
    Edge enhancement preprocessing with support for 2D/3D data.
    
    Args:
        strength: Strength of edge enhancement
        method: Edge detection method ('sobel', 'laplacian')
        apply_to_3d: Whether to apply to 3D data
    """
    def apply_edge_enhancement_single(data):
        """Apply edge enhancement to single data tensor."""
        data_dim = detect_data_dimension(data)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            return apply_2d_operation_to_3d(data, lambda x: apply_edge_enhancement_2d(x, strength, method))
        else:
            return apply_edge_enhancement_2d(data, strength, method)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_edge_enhancement_single)
        return processed_data, label
    
    return wrapper

def apply_edge_enhancement_2d(data, strength, method):
    """Apply edge enhancement to 2D data."""
    if method == 'sobel':
        # Sobel edge detection kernels
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        # Reshape for convolution
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Apply Sobel filters
        if len(data.shape) == 3:
            data_gray = tf.reduce_mean(data, axis=-1, keepdims=True)
            data_batch = tf.expand_dims(data_gray, 0)
        else:
            data_batch = data
            
        edges_x = tf.nn.conv2d(data_batch, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y = tf.nn.conv2d(data_batch, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        # Combine edges
        edges = tf.sqrt(edges_x**2 + edges_y**2)
        
        # Enhance original image with edges
        if len(data.shape) == 3:
            edges = tf.squeeze(edges, 0)
            enhanced = data + strength * tf.broadcast_to(edges, tf.shape(data))
        else:
            enhanced = data + strength * edges
            
    elif method == 'laplacian':
        # Laplacian kernel
        laplacian = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tf.float32)
        laplacian = tf.reshape(laplacian, [3, 3, 1, 1])
        
        if len(data.shape) == 3:
            data_gray = tf.reduce_mean(data, axis=-1, keepdims=True)
            data_batch = tf.expand_dims(data_gray, 0)
        else:
            data_batch = data
            
        edges = tf.nn.conv2d(data_batch, laplacian, strides=[1, 1, 1, 1], padding='SAME')
        
        if len(data.shape) == 3:
            edges = tf.squeeze(edges, 0)
            enhanced = data + strength * tf.broadcast_to(edges, tf.shape(data))
        else:
            enhanced = data + strength * edges
    else:
        enhanced = data
    
    return tf.clip_by_value(enhanced, 0.0, 1.0)

def intensity_windowing(window_min=0.0, window_max=1.0, apply_to_3d=True):
    """
    Intensity windowing (common in medical imaging) with 2D/3D support.
    
    Args:
        window_min: Minimum intensity value for windowing
        window_max: Maximum intensity value for windowing
        apply_to_3d: Whether to apply to 3D data
    """
    def apply_windowing_single(data):
        """Apply intensity windowing to single data tensor."""
        # Apply windowing
        windowed = tf.clip_by_value(data, window_min, window_max)
        
        # Normalize to [0, 1] range
        windowed = (windowed - window_min) / (window_max - window_min + 1e-8)
        
        return windowed
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_windowing_single)
        return processed_data, label
    
    return wrapper

def label_preprocessing(num_classes=None, task_type='classification', one_hot=True):
    """
    Label preprocessing for different task types.
    
    Args:
        num_classes: Number of classes (for one-hot encoding)
        task_type: Type of task ('classification', 'segmentation', 'object_detection')
        one_hot: Whether to apply one-hot encoding
    """
    def wrapper(data, label):
        # Infer task type if not provided
        if isinstance(label, (list, tuple)):
            # Multi-output case
            processed_labels = []
            for lbl in label:
                processed_lbl = process_single_label(lbl, num_classes, task_type, one_hot)
                processed_labels.append(processed_lbl)
            processed_label = processed_labels
        elif isinstance(label, dict):
            # Dict case
            processed_label = {}
            for key, lbl in label.items():
                processed_label[key] = process_single_label(lbl, num_classes, task_type, one_hot)
        else:
            # Single label case
            processed_label = process_single_label(label, num_classes, task_type, one_hot)
        
        return data, processed_label
    
    return wrapper

def process_single_label(label, num_classes, task_type, one_hot):
    """Process a single label tensor."""
    if task_type == 'classification' and one_hot and num_classes:
        # Convert to one-hot encoding
        if len(label.shape) == 0:  # Scalar
            return tf.one_hot(label, num_classes)
        elif len(label.shape) == 1 and label.shape[0] == 1:  # Single value
            return tf.one_hot(label[0], num_classes)
        else:
            return label  # Already one-hot or multi-dimensional
    elif task_type == 'segmentation':
        # For segmentation, ensure proper shape and data type
        if one_hot and num_classes:
            return tf.one_hot(tf.cast(label, tf.int32), num_classes)
        else:
            return tf.cast(label, tf.int32)
    else:
        return label

# Backward compatibility
adaptive_histogram_equalization = enhanced_adaptive_histogram_equalization


def edge_enhancement(strength=1.5, blur_radius=3):
    """
    Enhance edges in images using unsharp masking technique.
    
    Args:
        strength (float): Edge enhancement strength (higher = more enhancement) (default: 1.5)
        blur_radius (int): Gaussian blur radius for unsharp mask (default: 3)
    """
    def wrapper(data, label):
        if tf.is_tensor(data):
            # TensorFlow implementation
            kernel_size = blur_radius * 2 + 1
            
            # Create Gaussian kernel
            sigma = tf.cast(blur_radius, tf.float32)
            x = tf.range(-blur_radius, blur_radius + 1, dtype=tf.float32)
            kernel_1d = tf.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
            
            # Create 2D kernel
            kernel_2d = tf.outer(kernel_1d, kernel_1d)
            kernel_2d = tf.expand_dims(tf.expand_dims(kernel_2d, -1), -1)
            
            # Apply convolution for blur
            data_expanded = tf.expand_dims(data, 0)
            channels = tf.shape(data)[-1]
            kernel = tf.tile(kernel_2d, [1, 1, channels, 1])
            
            blurred = tf.nn.depthwise_conv2d(
                data_expanded,
                kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )[0]
            
            # Create unsharp mask and apply enhancement
            mask = data - blurred
            enhanced = data + strength * mask
            enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
            
        else:
            # NumPy implementation
            if len(data.shape) == 2:  # Grayscale
                blurred = cv2.GaussianBlur(data.astype(np.float32), 
                                         (blur_radius*2+1, blur_radius*2+1), 0)
                mask = data.astype(np.float32) - blurred
                enhanced = data.astype(np.float32) + strength * mask
                enhanced = np.clip(enhanced, 0, 255 if data.max() > 1 else 1).astype(data.dtype)
                
            elif len(data.shape) == 3:  # Color image
                enhanced = np.zeros_like(data, dtype=np.float32)
                for i in range(data.shape[2]):
                    channel = data[:, :, i].astype(np.float32)
                    blurred = cv2.GaussianBlur(channel, (blur_radius*2+1, blur_radius*2+1), 0)
                    mask = channel - blurred
                    enhanced[:, :, i] = channel + strength * mask
                max_val = 255 if data.max() > 1 else 1
                enhanced = np.clip(enhanced, 0, max_val).astype(data.dtype)
            else:
                enhanced = data
        
        return enhanced, label
    
    return wrapper


def gamma_correction(gamma=1.2, gain=1.0):
    """
    Apply gamma correction to adjust image brightness and contrast.
    
    Args:
        gamma (float): Gamma value (>1 darkens, <1 brightens) (default: 1.2)
        gain (float): Gain factor applied before gamma correction (default: 1.0)
    """
    def wrapper(data, label):
        if tf.is_tensor(data):
            # TensorFlow implementation
            # Ensure data is in 0-1 range
            normalized_data = data
            if tf.reduce_max(data) > 1.0:
                normalized_data = data / 255.0
                
            # Apply gamma correction: output = gain * input^gamma
            corrected = gain * tf.pow(normalized_data, gamma)
            corrected = tf.clip_by_value(corrected, 0.0, 1.0)
            
            # Convert back to original range if needed
            if tf.reduce_max(data) > 1.0:
                corrected = corrected * 255.0
                
        else:
            # NumPy implementation
            # Normalize to 0-1 range if needed
            if data.max() > 1.0:
                normalized = data.astype(np.float32) / 255.0
                was_uint8 = True
            else:
                normalized = data.astype(np.float32)
                was_uint8 = False
            
            # Apply gamma correction: output = gain * input^gamma
            corrected = gain * np.power(normalized, gamma)
            corrected = np.clip(corrected, 0.0, 1.0)
            
            # Convert back to original range
            if was_uint8:
                corrected = (corrected * 255).astype(data.dtype)
            else:
                corrected = corrected.astype(data.dtype)
                
        return corrected, label
    
    return wrapper


def normalize_custom(mean_values=[0.485, 0.456, 0.406], std_values=[0.229, 0.224, 0.225]):
    """
    Apply custom normalization with specified mean and standard deviation values.
    
    Args:
        mean_values (list): Mean values for each channel (default: ImageNet means)
        std_values (list): Standard deviation values for each channel (default: ImageNet stds)
    """
    def wrapper(data, label):
        if tf.is_tensor(data):
            # TensorFlow implementation
            mean_tensor = tf.constant(mean_values, dtype=data.dtype)
            std_tensor = tf.constant(std_values, dtype=data.dtype)
            
            # Ensure data is in 0-1 range
            normalized_data = data
            if tf.reduce_max(data) > 1.0:
                normalized_data = data / 255.0
                
            # Apply normalization: (x - mean) / std
            normalized = (normalized_data - mean_tensor) / std_tensor
            
        else:
            # NumPy implementation
            mean_array = np.array(mean_values, dtype=data.dtype)
            std_array = np.array(std_values, dtype=data.dtype)
            
            # Ensure data is in 0-1 range
            if data.max() > 1.0:
                data_normalized = data.astype(np.float32) / 255.0
            else:
                data_normalized = data.astype(np.float32)
                
            # Apply normalization: (x - mean) / std
            normalized = (data_normalized - mean_array) / std_array
            normalized = normalized.astype(data.dtype)
            
        return normalized, label
    
    return wrapper


def resize_with_pad(target_height=224, target_width=224, pad_value=0):
    """
    Resize image while maintaining aspect ratio and pad to target size.
    
    Args:
        target_height (int): Target height (default: 224)
        target_width (int): Target width (default: 224)  
        pad_value (float): Value to use for padding (default: 0)
    """
    def wrapper(data, label):
        if tf.is_tensor(data):
            # TensorFlow implementation
            resized = tf.image.resize_with_pad(
                data,
                target_height,
                target_width,
                method=tf.image.ResizeMethod.BILINEAR,
                antialias=False
            )
        else:
            # NumPy implementation with OpenCV
            h, w = data.shape[:2]
            scale = min(target_height / h, target_width / w)
            
            new_h, new_w = int(h * scale), int(w * scale)
            resized_data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create padded image
            if len(data.shape) == 3:
                padded = np.full((target_height, target_width, data.shape[2]), pad_value, dtype=data.dtype)
            else:
                padded = np.full((target_height, target_width), pad_value, dtype=data.dtype)
                
            # Center the resized image
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            
            if len(data.shape) == 3:
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_data
            else:
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_data
                
            resized = padded
            
        return resized, label
    
    return wrapper


def volume_slice_normalization(slice_wise=True, method='z_score'):
    """
    Apply normalization to 3D volumes on a slice-by-slice basis or globally.
    Compatible with both 2D images and 3D volumes.
    
    Args:
        slice_wise (bool): If True, normalize each slice separately. If False, normalize globally (default: True)
        method (str): Normalization method - 'z_score', 'min_max', or 'robust' (default: 'z_score')
    """
    def wrapper(data, label):
        if tf.is_tensor(data):
            data_tensor = data
        else:
            data_tensor = tf.constant(data)
            
        original_shape = tf.shape(data_tensor)
        
        # Check if data is 3D (has depth dimension)
        if len(original_shape) >= 4:  # [batch, depth, height, width, channels] or similar
            if slice_wise:
                # Process each slice separately
                processed_slices = []
                depth_dim = original_shape[1] if len(original_shape) == 5 else original_shape[0]
                
                for i in range(depth_dim):
                    if len(original_shape) == 5:  # [batch, depth, height, width, channels]
                        slice_data = data_tensor[:, i, :, :, :]
                    else:  # [depth, height, width, channels]
                        slice_data = data_tensor[i, :, :, :]
                    
                    # Apply normalization to this slice
                    if method == 'z_score':
                        mean_val = tf.reduce_mean(slice_data)
                        std_val = tf.math.reduce_std(slice_data)
                        std_val = tf.maximum(std_val, 1e-8)  # Avoid division by zero
                        normalized_slice = (slice_data - mean_val) / std_val
                    elif method == 'min_max':
                        min_val = tf.reduce_min(slice_data)
                        max_val = tf.reduce_max(slice_data)
                        range_val = tf.maximum(max_val - min_val, 1e-8)
                        normalized_slice = (slice_data - min_val) / range_val
                    elif method == 'robust':
                        # Use percentiles for robust normalization
                        flattened = tf.reshape(slice_data, [-1])
                        sorted_data = tf.sort(flattened)
                        n = tf.shape(sorted_data)[0]
                        q25 = sorted_data[n // 4]
                        q75 = sorted_data[3 * n // 4]
                        median = sorted_data[n // 2]
                        iqr = tf.maximum(q75 - q25, 1e-8)
                        normalized_slice = (slice_data - median) / iqr
                    else:
                        normalized_slice = slice_data
                    
                    processed_slices.append(normalized_slice)
                
                # Stack slices back together
                if len(original_shape) == 5:
                    normalized = tf.stack(processed_slices, axis=1)
                else:
                    normalized = tf.stack(processed_slices, axis=0)
            else:
                # Global normalization across entire volume
                if method == 'z_score':
                    mean_val = tf.reduce_mean(data_tensor)
                    std_val = tf.math.reduce_std(data_tensor)
                    std_val = tf.maximum(std_val, 1e-8)
                    normalized = (data_tensor - mean_val) / std_val
                elif method == 'min_max':
                    min_val = tf.reduce_min(data_tensor)
                    max_val = tf.reduce_max(data_tensor)
                    range_val = tf.maximum(max_val - min_val, 1e-8)
                    normalized = (data_tensor - min_val) / range_val
                elif method == 'robust':
                    flattened = tf.reshape(data_tensor, [-1])
                    sorted_data = tf.sort(flattened)
                    n = tf.shape(sorted_data)[0]
                    q25 = sorted_data[n // 4]
                    q75 = sorted_data[3 * n // 4]
                    median = sorted_data[n // 2]
                    iqr = tf.maximum(q75 - q25, 1e-8)
                    normalized = (data_tensor - median) / iqr
                else:
                    normalized = data_tensor
        else:
            # 2D data - apply normalization globally
            if method == 'z_score':
                mean_val = tf.reduce_mean(data_tensor)
                std_val = tf.math.reduce_std(data_tensor)
                std_val = tf.maximum(std_val, 1e-8)
                normalized = (data_tensor - mean_val) / std_val
            elif method == 'min_max':
                min_val = tf.reduce_min(data_tensor)
                max_val = tf.reduce_max(data_tensor)
                range_val = tf.maximum(max_val - min_val, 1e-8)
                normalized = (data_tensor - min_val) / range_val
            elif method == 'robust':
                flattened = tf.reshape(data_tensor, [-1])
                sorted_data = tf.sort(flattened)
                n = tf.shape(sorted_data)[0]
                q25 = sorted_data[n // 4]
                q75 = sorted_data[3 * n // 4]
                median = sorted_data[n // 2]
                iqr = tf.maximum(q75 - q25, 1e-8)
                normalized = (data_tensor - median) / iqr
            else:
                normalized = data_tensor
        
        # Convert back to original type if needed
        if not tf.is_tensor(data):
            normalized = normalized.numpy()
            
        return normalized, label
    
    return wrapper
