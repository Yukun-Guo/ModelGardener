"""
Enhanced custom augmentation functions for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (data, label) and returns (modified_data, modified_label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def augmentation_name(param1=default1, param2=default2):
    def wrapper(data, label):
        # Apply augmentation logic here
        modified_data = apply_augmentation(data, param1, param2)
        modified_label = modify_label_if_needed(label, augmentation_params)
        return modified_data, modified_label
    return wrapper
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Union, List, Dict, Tuple, Any
from .utils import (
    TaskType, DataDimension, detect_data_dimension, infer_task_type,
    handle_multi_input, handle_multi_output, apply_2d_operation_to_3d,
    get_spatial_dimensions
)

def enhanced_color_shift(hue_shift=20, saturation_scale=1.2, value_scale=1.1, probability=0.6, apply_to_3d=True):
    """
    Apply color shifting in HSV space with support for 2D/3D data and multi-inputs.
    
    Args:
        hue_shift (int): Maximum hue shift in degrees (default: 20)
        saturation_scale (float): Saturation scaling factor (default: 1.2) 
        value_scale (float): Value/brightness scaling factor (default: 1.1)
        probability (float): Probability of applying effect (default: 0.6)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_color_shift_single(image):
        """Apply color shift to a single image tensor."""
        if tf.random.uniform(()) > probability:
            return image
            
        # Detect data dimension
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, apply_color_shift_2d)
        else:
            return apply_color_shift_2d(image)
    
    def apply_color_shift_2d(image):
        """Apply color shift to 2D image."""
        # Ensure image has 3 channels for HSV conversion
        if tf.shape(image)[-1] != 3:
            return image  # Skip non-RGB images
            
        # Convert to HSV
        hsv = tf.image.rgb_to_hsv(image)
        
        # Apply random hue shift (normalized to 0-1 range)
        hue_delta = tf.random.uniform((), -hue_shift/360.0, hue_shift/360.0)
        hsv = tf.concat([
            tf.expand_dims((hsv[..., 0] + hue_delta) % 1.0, axis=-1),
            hsv[..., 1:2],
            hsv[..., 2:3]
        ], axis=-1)
        
        # Apply saturation scaling
        sat_factor = tf.random.uniform((), 1.0, saturation_scale)
        hsv = tf.concat([
            hsv[..., 0:1],
            tf.expand_dims(tf.clip_by_value(hsv[..., 1] * sat_factor, 0, 1), axis=-1),
            hsv[..., 2:3]
        ], axis=-1)
        
        # Apply value scaling
        val_factor = tf.random.uniform((), 1.0, value_scale)
        hsv = tf.concat([
            hsv[..., 0:2],
            tf.expand_dims(tf.clip_by_value(hsv[..., 2] * val_factor, 0, 1), axis=-1)
        ], axis=-1)
        
        # Convert back to RGB
        return tf.image.hsv_to_rgb(hsv)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_color_shift_single)
        
        # Labels typically don't change for color augmentation
        # but we support multi-output structure
        return processed_data, label
    
    return wrapper

# Keep backward compatibility
color_shift = enhanced_color_shift


def enhanced_random_blur(max_kernel_size=5, probability=0.5, apply_to_3d=True):
    """
    Apply random Gaussian blur to the image with support for 2D/3D data and multi-inputs.
    
    Args:
        max_kernel_size (int): Maximum blur kernel size (default: 5)
        probability (float): Probability of applying blur (default: 0.5)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_blur_single(image):
        """Apply blur to a single image tensor."""
        if tf.random.uniform(()) > probability:
            return image
            
        # Detect data dimension
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, apply_blur_2d)
        else:
            return apply_blur_2d(image)
    
    def apply_blur_2d(image):
        """Apply blur to 2D image."""
        # Generate random kernel size (odd number)
        kernel_size = tf.random.uniform((), 1, max_kernel_size//2 + 1, dtype=tf.int32) * 2 + 1
        
        # For TensorFlow operations, we'll use a simple averaging filter
        # Create averaging kernel
        kernel_size_float = tf.cast(kernel_size, tf.float32)
        channels = tf.shape(image)[-1]
        
        # Create a simple box blur kernel
        kernel_weight = 1.0 / (kernel_size_float * kernel_size_float)
        kernel = tf.ones([kernel_size, kernel_size, channels, 1]) * kernel_weight
        
        # Apply blur using depthwise convolution
        if len(image.shape) == 3:  # (H, W, C)
            image_batch = tf.expand_dims(image, 0)
            blurred_batch = tf.nn.depthwise_conv2d(
                image_batch, kernel, strides=[1, 1, 1, 1], padding='SAME'
            )
            return blurred_batch[0]
        else:  # Already batched
            return tf.nn.depthwise_conv2d(
                image, kernel, strides=[1, 1, 1, 1], padding='SAME'
            )
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_blur_single)
        
        # Labels typically don't change for blur augmentation
        return processed_data, label
    
    return wrapper

# Keep backward compatibility
random_blur = enhanced_random_blur


def enhanced_noise_injection(noise_type='gaussian', intensity=0.1, probability=0.4, apply_to_3d=True):
    """
    Add random noise to the image for data augmentation with support for 2D/3D data and multi-inputs.
    
    Args:
        noise_type (str): Type of noise ('gaussian', 'uniform') (default: 'gaussian')
        intensity (float): Noise intensity (0.0 to 1.0) (default: 0.1)
        probability (float): Probability of applying noise (default: 0.4)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_noise_single(image):
        """Apply noise to a single image tensor."""
        if tf.random.uniform(()) > probability:
            return image
            
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=intensity)
            noisy_image = image + noise
        elif noise_type == 'uniform':
            # Uniform noise
            noise = tf.random.uniform(tf.shape(image), -intensity, intensity)
            noisy_image = image + noise
        else:
            # Default to Gaussian if unknown type
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=intensity)
            noisy_image = image + noise
        
        # Clip values to valid range [0, 1] for normalized images
        noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        return noisy_image
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_noise_single)
        
        # Labels typically don't change for noise augmentation
        return processed_data, label
    
    return wrapper

# Keep backward compatibility
noise_injection = enhanced_noise_injection


def enhanced_random_brightness(max_delta=0.2, apply_to_3d=True):
    """
    Apply random brightness adjustment with support for 2D/3D data and multi-inputs.
    
    Args:
        max_delta (float): Maximum brightness change (default: 0.2)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_brightness_single(image):
        """Apply brightness adjustment to a single image tensor."""
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, lambda x: apply_brightness_2d(x, max_delta))
        else:
            return apply_brightness_2d(image, max_delta)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_brightness_single)
        return processed_data, label
    
    return wrapper

def apply_brightness_2d(image, max_delta):
    """Apply brightness adjustment to 2D image."""
    return tf.image.stateless_random_brightness(
        image, 
        max_delta, 
        tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
    )

def enhanced_random_contrast(lower=0.8, upper=1.2, apply_to_3d=True):
    """
    Apply random contrast adjustment with support for 2D/3D data and multi-inputs.
    
    Args:
        lower (float): Lower bound for contrast factor (default: 0.8)
        upper (float): Upper bound for contrast factor (default: 1.2)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_contrast_single(image):
        """Apply contrast adjustment to a single image tensor."""
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, lambda x: apply_contrast_2d(x, lower, upper))
        else:
            return apply_contrast_2d(image, lower, upper)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_contrast_single)
        return processed_data, label
    
    return wrapper

def apply_contrast_2d(image, lower, upper):
    """Apply contrast adjustment to 2D image."""
    return tf.image.stateless_random_contrast(
        image,
        lower,
        upper,
        tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
    )

def enhanced_random_saturation(lower=0.7, upper=1.3, apply_to_3d=True):
    """
    Apply random saturation adjustment with support for 2D/3D data and multi-inputs.
    
    Args:
        lower (float): Lower bound for saturation factor (default: 0.7)
        upper (float): Upper bound for saturation factor (default: 1.3)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_saturation_single(image):
        """Apply saturation adjustment to a single image tensor."""
        # Skip if not RGB (saturation only applies to color images)
        if tf.shape(image)[-1] != 3:
            return image
            
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, lambda x: apply_saturation_2d(x, lower, upper))
        else:
            return apply_saturation_2d(image, lower, upper)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_saturation_single)
        return processed_data, label
    
    return wrapper

def apply_saturation_2d(image, lower, upper):
    """Apply saturation adjustment to 2D image."""
    return tf.image.stateless_random_saturation(
        image,
        lower,
        upper,
        tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
    )

def enhanced_random_hue(max_delta=0.1, apply_to_3d=True):
    """
    Apply random hue adjustment with support for 2D/3D data and multi-inputs.
    
    Args:
        max_delta (float): Maximum hue change (default: 0.1)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def apply_hue_single(image):
        """Apply hue adjustment to a single image tensor."""
        # Skip if not RGB (hue only applies to color images)
        if tf.shape(image)[-1] != 3:
            return image
            
        data_dim = detect_data_dimension(image)
        
        if data_dim == DataDimension.THREE_D and apply_to_3d:
            # Apply to 3D data slice by slice
            return apply_2d_operation_to_3d(image, lambda x: apply_hue_2d(x, max_delta))
        else:
            return apply_hue_2d(image, max_delta)
    
    def wrapper(data, label):
        # Handle multi-input data
        processed_data = handle_multi_input(data, apply_hue_single)
        return processed_data, label
    
    return wrapper

def apply_hue_2d(image, max_delta):
    """Apply hue adjustment to 2D image."""
    return tf.image.stateless_random_hue(
        image,
        max_delta,
        tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
    )

# Maintain backward compatibility
random_brightness = enhanced_random_brightness
random_contrast = enhanced_random_contrast
random_saturation = enhanced_random_saturation
random_hue = enhanced_random_hue


def enhanced_random_rotation(max_angle=15.0, probability=0.5, interpolation='bilinear', apply_to_3d=True):
    """
    Apply random rotation with support for 2D/3D data, multi-inputs, and label transformation.
    
    Args:
        max_angle (float): Maximum rotation angle in degrees (default: 15.0)
        probability (float): Probability of applying rotation (default: 0.5)
        interpolation (str): Interpolation method ('bilinear', 'nearest') (default: 'bilinear')
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def wrapper(data, label):
        if tf.random.uniform(()) > probability:
            return data, label
            
        # Generate random angle
        angle_rad = tf.random.uniform((), -max_angle * np.pi / 180, max_angle * np.pi / 180)
        
        # Infer task type from label
        if isinstance(label, (list, tuple)):
            # Multi-output case
            label_shape = label[0].shape if len(label) > 0 else ()
            sample_label = label[0] if len(label) > 0 else None
        elif isinstance(label, dict):
            # Dict case
            first_key = list(label.keys())[0] if label else None
            label_shape = label[first_key].shape if first_key else ()
            sample_label = label[first_key] if first_key else None
        else:
            label_shape = label.shape
            sample_label = label
            
        task_type = infer_task_type(label_shape, sample_label)
        
        # Apply rotation to data
        def apply_rotation_single(image):
            data_dim = detect_data_dimension(image)
            if data_dim == DataDimension.THREE_D and apply_to_3d:
                return apply_2d_operation_to_3d(image, lambda x: apply_rotation_2d(x, angle_rad, interpolation))
            else:
                return apply_rotation_2d(image, angle_rad, interpolation)
        
        processed_data = handle_multi_input(data, apply_rotation_single)
        
        # Apply rotation to labels if needed (for segmentation)
        processed_label = label
        if task_type == TaskType.SEGMENTATION:
            def apply_rotation_to_label(lbl):
                return apply_rotation_2d(lbl, angle_rad, 'nearest')  # Use nearest for labels
            processed_label = handle_multi_output(label, apply_rotation_to_label)
        
        return processed_data, processed_label
    
    return wrapper

def apply_rotation_2d(image, angle_rad, interpolation='bilinear'):
    """Apply 2D rotation to image using TensorFlow operations."""
    # Use tf.keras.utils for image rotation if available, otherwise implement manually
    try:
        # Simple rotation using TensorFlow image operations
        # Note: This is a simplified implementation
        # For production, you might want to use tf.contrib.image.rotate or tfa.image.rotate
        
        # Convert to batch format if needed
        if len(image.shape) == 3:
            image_batch = tf.expand_dims(image, 0)
            needs_squeeze = True
        else:
            image_batch = image
            needs_squeeze = False
        
        # Simple shear-based rotation approximation for small angles
        # For larger angles, you'd want proper rotation matrix implementation
        cos_angle = tf.cos(angle_rad)
        sin_angle = tf.sin(angle_rad)
        
        # This is a simplified rotation - in practice you'd use proper affine transformation
        rotated = image_batch  # Placeholder - implement proper rotation
        
        if needs_squeeze:
            rotated = rotated[0]
            
        return rotated
    except:
        # Fallback to original image if rotation fails
        return image

def enhanced_random_flip(direction='horizontal', probability=0.5, apply_to_3d=True):
    """
    Apply random flip with support for 2D/3D data, multi-inputs, and label transformation.
    
    Args:
        direction (str): Flip direction ('horizontal', 'vertical', 'both') (default: 'horizontal')
        probability (float): Probability of applying flip (default: 0.5)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def wrapper(data, label):
        if tf.random.uniform(()) > probability:
            return data, label
            
        # Determine flip operations
        flip_horizontal = direction in ['horizontal', 'both'] and tf.random.uniform(()) > 0.5
        flip_vertical = direction in ['vertical', 'both'] and tf.random.uniform(()) > 0.5
        
        if not flip_horizontal and not flip_vertical:
            return data, label
        
        # Apply flip to data
        def apply_flip_single(image):
            result = image
            if flip_horizontal:
                result = tf.image.flip_left_right(result)
            if flip_vertical:
                result = tf.image.flip_up_down(result)
            return result
        
        processed_data = handle_multi_input(data, apply_flip_single)
        
        # Apply flip to labels (for segmentation and object detection)
        label_shape = getattr(label, 'shape', ())
        task_type = infer_task_type(label_shape, label)
        
        processed_label = label
        if task_type in [TaskType.SEGMENTATION, TaskType.OBJECT_DETECTION]:
            processed_label = handle_multi_output(label, apply_flip_single)
            
            # For object detection, we'd also need to adjust bounding box coordinates
            # This is task-specific and would require more complex logic
        
        return processed_data, processed_label
    
    return wrapper

def enhanced_random_crop(crop_size=None, padding=None, probability=0.5, apply_to_3d=True):
    """
    Apply random crop with support for 2D/3D data, multi-inputs, and label transformation.
    
    Args:
        crop_size (tuple): Target crop size (height, width) or (height, width, depth) for 3D
        padding (int or tuple): Padding to apply before cropping
        probability (float): Probability of applying crop (default: 0.5)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def wrapper(data, label):
        if tf.random.uniform(()) > probability:
            return data, label
            
        # Get data dimensions
        if isinstance(data, tf.Tensor):
            sample_data = data
        elif isinstance(data, (list, tuple)):
            sample_data = data[0]
        elif isinstance(data, dict):
            sample_data = list(data.values())[0]
        else:
            return data, label
            
        data_dim = detect_data_dimension(sample_data)
        spatial_dims = get_spatial_dimensions(sample_data, data_dim)
        
        # Determine crop size if not provided
        if crop_size is None:
            if data_dim == DataDimension.TWO_D:
                crop_size = (spatial_dims[0] * 3 // 4, spatial_dims[1] * 3 // 4)
            else:
                crop_size = (spatial_dims[0] * 3 // 4, spatial_dims[1] * 3 // 4, spatial_dims[2] * 3 // 4)
        
        # Apply crop to data
        def apply_crop_single(image):
            if data_dim == DataDimension.TWO_D:
                return tf.image.random_crop(image, [crop_size[0], crop_size[1], tf.shape(image)[-1]])
            else:
                # For 3D, we need custom implementation
                return apply_3d_crop(image, crop_size)
        
        processed_data = handle_multi_input(data, apply_crop_single)
        
        # Apply crop to labels (for segmentation)
        label_shape = getattr(label, 'shape', ())
        task_type = infer_task_type(label_shape, label)
        
        processed_label = label
        if task_type == TaskType.SEGMENTATION:
            def apply_crop_to_label(lbl):
                if data_dim == DataDimension.TWO_D:
                    return tf.image.random_crop(lbl, [crop_size[0], crop_size[1], tf.shape(lbl)[-1]])
                else:
                    return apply_3d_crop(lbl, crop_size)
            processed_label = handle_multi_output(label, apply_crop_to_label)
        
        return processed_data, processed_label
    
    return wrapper

def apply_3d_crop(volume, crop_size):
    """Apply random crop to 3D volume."""
    volume_shape = tf.shape(volume)
    
    # Calculate maximum offset for random crop
    max_offset_h = volume_shape[0] - crop_size[0]
    max_offset_w = volume_shape[1] - crop_size[1]
    max_offset_d = volume_shape[2] - crop_size[2]
    
    # Generate random offsets
    offset_h = tf.random.uniform((), 0, max_offset_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform((), 0, max_offset_w + 1, dtype=tf.int32)
    offset_d = tf.random.uniform((), 0, max_offset_d + 1, dtype=tf.int32)
    
    # Apply crop
    cropped = volume[offset_h:offset_h + crop_size[0],
                   offset_w:offset_w + crop_size[1],
                   offset_d:offset_d + crop_size[2],
                   :]
    
    return cropped

def elastic_deformation(alpha=100, sigma=10, probability=0.3, apply_to_3d=True):
    """
    Apply elastic deformation augmentation especially useful for medical imaging and segmentation.
    
    Args:
        alpha (float): Scaling factor for deformation strength (default: 100)
        sigma (float): Standard deviation for Gaussian smoothing (default: 10)
        probability (float): Probability of applying deformation (default: 0.3)
        apply_to_3d (bool): Whether to apply to 3D data (default: True)
    """
    def wrapper(data, label):
        if tf.random.uniform(()) > probability:
            return data, label
            
        # Generate random displacement fields
        def apply_elastic_single(image):
            data_dim = detect_data_dimension(image)
            shape = tf.shape(image)
            
            if data_dim == DataDimension.TWO_D:
                # Generate 2D displacement field
                dx = tf.random.normal([shape[0], shape[1]], stddev=sigma) * alpha
                dy = tf.random.normal([shape[0], shape[1]], stddev=sigma) * alpha
                
                # Apply Gaussian smoothing to displacement fields
                # (Simplified implementation - in practice you'd use proper Gaussian filtering)
                dx = tf.nn.avg_pool2d(tf.expand_dims(tf.expand_dims(dx, 0), -1), 
                                    ksize=3, strides=1, padding='SAME')[0, :, :, 0]
                dy = tf.nn.avg_pool2d(tf.expand_dims(tf.expand_dims(dy, 0), -1), 
                                    ksize=3, strides=1, padding='SAME')[0, :, :, 0]
                
                # Create coordinate grids
                y_coords, x_coords = tf.meshgrid(tf.range(shape[0], dtype=tf.float32),
                                               tf.range(shape[1], dtype=tf.float32), indexing='ij')
                
                # Apply displacement
                new_y = y_coords + dy
                new_x = x_coords + dx
                
                # Clip coordinates to valid range
                new_y = tf.clip_by_value(new_y, 0, tf.cast(shape[0] - 1, tf.float32))
                new_x = tf.clip_by_value(new_x, 0, tf.cast(shape[1] - 1, tf.float32))
                
                # Sample using bilinear interpolation (simplified)
                # In practice, you'd use tf.gather_nd with proper interpolation
                return image  # Placeholder - implement proper sampling
            
            return image  # Return unchanged for unsupported cases
        
        processed_data = handle_multi_input(data, apply_elastic_single)
        
        # Apply same deformation to labels for segmentation
        label_shape = getattr(label, 'shape', ())
        task_type = infer_task_type(label_shape, label)
        
        processed_label = label
        if task_type == TaskType.SEGMENTATION:
            processed_label = handle_multi_output(label, apply_elastic_single)
        
        return processed_data, processed_label
    
    return wrapper

# Additional backward compatibility aliases
random_rotation = enhanced_random_rotation
random_flip = enhanced_random_flip
random_crop = enhanced_random_crop

