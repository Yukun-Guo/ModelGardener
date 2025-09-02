"""
Custom Preprocessing Template for ModelGardener

This file provides templates for creating custom preprocessing functions.
Preprocessing functions should work with individual samples or batches.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List


def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image using ImageNet statistics or custom values.
    
    Args:
        image: Input image tensor (0-1 range)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized image
    """
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    
    # Normalize
    image = (image - mean) / std
    
    return image


def resize_with_padding(image, target_size, pad_value=0):
    """
    Resize image while maintaining aspect ratio using padding.
    
    Args:
        image: Input image tensor
        target_size: Target size [height, width]
        pad_value: Value to use for padding
        
    Returns:
        Resized and padded image
    """
    target_height, target_width = target_size
    
    # Get original dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Calculate scaling factor
    scale = tf.minimum(
        tf.cast(target_height, tf.float32) / tf.cast(height, tf.float32),
        tf.cast(target_width, tf.float32) / tf.cast(width, tf.float32)
    )
    
    # Calculate new dimensions
    new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
    
    # Resize image
    image = tf.image.resize(image, [new_height, new_width])
    
    # Calculate padding
    pad_height = target_height - new_height
    pad_width = target_width - new_width
    
    # Apply padding
    paddings = [
        [pad_height // 2, pad_height - pad_height // 2],
        [pad_width // 2, pad_width - pad_width // 2],
        [0, 0]
    ]
    
    image = tf.pad(image, paddings, constant_values=pad_value)
    
    return image


def histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: Input image tensor (0-1 range)
        
    Returns:
        Equalized image
    """
    # Convert to uint8
    image_uint8 = tf.cast(image * 255, tf.uint8)
    
    # Apply histogram equalization per channel
    channels = []
    for c in range(tf.shape(image)[-1]):
        channel = image_uint8[:, :, c]
        # Flatten and compute histogram
        channel_flat = tf.reshape(channel, [-1])
        hist = tf.histogram_fixed_width(tf.cast(channel_flat, tf.float32), [0.0, 255.0], nbins=256)
        
        # Compute CDF
        cdf = tf.cumsum(hist)
        cdf_normalized = cdf / tf.cast(tf.reduce_max(cdf), tf.float32) * 255
        
        # Apply equalization
        channel_eq = tf.gather(cdf_normalized, channel)
        channels.append(channel_eq)
    
    # Combine channels
    image_eq = tf.stack(channels, axis=-1)
    image_eq = tf.cast(image_eq, tf.float32) / 255.0
    
    return image_eq


def apply_gaussian_noise(image, stddev=0.1):
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image tensor
        stddev: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = tf.random.normal(tf.shape(image), stddev=stddev)
    noisy_image = image + noise
    
    # Clip to valid range
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    
    return noisy_image


def edge_enhancement(image, kernel_size=3, sigma=1.0):
    """
    Apply edge enhancement using Gaussian derivatives.
    
    Args:
        image: Input image tensor
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of Gaussian
        
    Returns:
        Edge-enhanced image
    """
    # Convert to grayscale if needed
    if tf.shape(image)[-1] == 3:
        gray = tf.reduce_mean(image, axis=-1, keepdims=True)
    else:
        gray = image
    
    # Create Gaussian kernel
    kernel = tf.cast(tf.range(kernel_size), tf.float32) - (kernel_size - 1) / 2
    kernel = tf.exp(-0.5 * tf.square(kernel) / (sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Reshape for conv2d
    kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1])
    
    # Apply horizontal and vertical gradients
    grad_x = tf.nn.conv2d(gray, kernel, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(gray, tf.transpose(kernel, [1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
    
    # Compute gradient magnitude
    gradient_mag = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))
    
    # Enhance edges
    enhanced = image + 0.5 * gradient_mag
    enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
    
    return enhanced


def color_space_conversion(image, from_space='RGB', to_space='HSV'):
    """
    Convert image between color spaces.
    
    Args:
        image: Input image tensor
        from_space: Source color space
        to_space: Target color space
        
    Returns:
        Converted image
    """
    if from_space == 'RGB' and to_space == 'HSV':
        return tf.image.rgb_to_hsv(image)
    elif from_space == 'HSV' and to_space == 'RGB':
        return tf.image.hsv_to_rgb(image)
    elif from_space == 'RGB' and to_space == 'LAB':
        # Simplified RGB to LAB conversion
        # Note: This is a simplified version, full LAB conversion is more complex
        return tf.image.rgb_to_yuv(image)
    else:
        return image


def adaptive_preprocessing(image, image_stats=None):
    """
    Apply adaptive preprocessing based on image statistics.
    
    Args:
        image: Input image tensor
        image_stats: Precomputed image statistics
        
    Returns:
        Preprocessed image
    """
    if image_stats is None:
        # Compute statistics
        mean_val = tf.reduce_mean(image)
        std_val = tf.math.reduce_std(image)
        brightness = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
    else:
        mean_val, std_val, brightness = image_stats
    
    # Adaptive normalization
    if std_val > 0.2:
        # High variance - apply histogram equalization
        image = histogram_equalization(image)
    
    if brightness < 0.3:
        # Dark image - apply gamma correction
        gamma = 0.7
        image = tf.pow(image, gamma)
    elif brightness > 0.7:
        # Bright image - reduce brightness
        image = image * 0.9
    
    # Final normalization
    image = normalize_image(image)
    
    return image


class PreprocessingPipeline:
    """
    Class for creating preprocessing pipelines.
    """
    
    def __init__(self, steps, apply_order='sequential'):
        """
        Args:
            steps: List of preprocessing functions
            apply_order: Order to apply steps ('sequential' or 'random')
        """
        self.steps = steps
        self.apply_order = apply_order
    
    def __call__(self, image, label=None):
        """Apply preprocessing steps."""
        if self.apply_order == 'random':
            # Apply steps in random order
            steps = tf.random.shuffle(self.steps)
        else:
            steps = self.steps
        
        for step in steps:
            try:
                image = step(image)
            except Exception as e:
                print(f"Warning: Preprocessing step failed: {e}")
                continue
        
        return (image, label) if label is not None else image


# Batch preprocessing functions
def batch_normalize(batch_images, batch_size=None):
    """
    Normalize a batch of images.
    
    Args:
        batch_images: Batch of images [batch, height, width, channels]
        batch_size: Size of batch
        
    Returns:
        Normalized batch
    """
    # Compute batch statistics
    batch_mean = tf.reduce_mean(batch_images, axis=[1, 2, 3], keepdims=True)
    batch_var = tf.math.reduce_variance(batch_images, axis=[1, 2, 3], keepdims=True)
    
    # Normalize
    normalized_batch = (batch_images - batch_mean) / (tf.sqrt(batch_var) + 1e-8)
    
    return normalized_batch


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom preprocessing functions...")
    
    # Create dummy image
    dummy_image = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
    
    # Test normalization
    norm_image = normalize_image(dummy_image)
    print(f"Normalized image shape: {norm_image.shape}")
    
    # Test resize with padding
    padded_image = resize_with_padding(dummy_image, [256, 256])
    print(f"Padded image shape: {padded_image.shape}")
    
    # Test preprocessing pipeline
    pipeline = PreprocessingPipeline([
        normalize_image,
        lambda img: apply_gaussian_noise(img, 0.05)
    ])
    processed_image = pipeline(dummy_image)
    print(f"Pipeline processed shape: {processed_image.shape}")
    
    print("✅ Custom preprocessing template ready!")
