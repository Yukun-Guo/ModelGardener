"""
Example custom augmentation functions for ModelGardener.

All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (image, label) and returns (modified_image, label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def augmentation_name(param1=default1, param2=default2):
    def wrapper(image, label):
        # Apply augmentation logic here
        modified_image = apply_augmentation(image, param1, param2)
        return modified_image, label
    return wrapper
"""

import numpy as np
import cv2
import tensorflow as tf

def tf_color_shift(hue_shift=20, saturation_scale=1.2, value_scale=1.1, probability=0.6):
    """
    Apply color shifting in HSV space.
    
    Args:
        hue_shift (int): Maximum hue shift in degrees (default: 20)
        saturation_scale (float): Saturation scaling factor (default: 1.2) 
        value_scale (float): Value/brightness scaling factor (default: 1.1)
        probability (float): Probability of applying effect (default: 0.6)
    """
    def wrapper(image, label):
        if tf.random.uniform(()) > probability:
            return image, label
            
        # Convert to HSV
        hsv = tf.image.rgb_to_hsv(image)
        
        # Apply random hue shift (normalized to 0-1 range)
        hue_delta = tf.random.uniform((), -hue_shift/360.0, hue_shift/360.0)
        hsv = tf.concat([
            tf.expand_dims((hsv[:, :, 0] + hue_delta) % 1.0, axis=-1),
            hsv[:, :, 1:2],
            hsv[:, :, 2:3]
        ], axis=-1)
        
        # Apply saturation scaling
        sat_factor = tf.random.uniform((), 1.0, saturation_scale)
        hsv = tf.concat([
            hsv[:, :, 0:1],
            tf.expand_dims(tf.clip_by_value(hsv[:, :, 1] * sat_factor, 0, 1), axis=-1),
            hsv[:, :, 2:3]
        ], axis=-1)
        
        # Apply value scaling
        val_factor = tf.random.uniform((), 1.0, value_scale)
        hsv = tf.concat([
            hsv[:, :, 0:2],
            tf.expand_dims(tf.clip_by_value(hsv[:, :, 2] * val_factor, 0, 1), axis=-1)
        ], axis=-1)
        
        # Convert back to RGB
        result = tf.image.hsv_to_rgb(hsv)
        return result, label
    
    return wrapper


def tf_random_blur(max_kernel_size=5, probability=0.5):
    """
    Apply random Gaussian blur to the image.
    
    Args:
        max_kernel_size (int): Maximum blur kernel size (default: 5)
        probability (float): Probability of applying blur (default: 0.5)
    """
    def wrapper(image, label):
        if tf.random.uniform(()) > probability:
            return image, label
            
        # Generate random kernel size (odd number)
        kernel_size = tf.random.uniform((), 1, max_kernel_size//2 + 1, dtype=tf.int32) * 2 + 1
        
        # Create Gaussian kernel
        sigma = tf.cast(kernel_size, tf.float32) / 3.0
        
        # Apply Gaussian blur using TensorFlow operations
        # Note: This is a simplified version - for full Gaussian blur, you'd need more complex ops
        blurred = tf.nn.depthwise_conv2d(
            tf.expand_dims(image, 0),
            tf.ones([kernel_size, kernel_size, tf.shape(image)[-1], 1]) / tf.cast(kernel_size * kernel_size, tf.float32),
            strides=[1, 1, 1, 1],
            padding='SAME'
        )[0]
        
        return blurred, label
    
    return wrapper


def tf_noise_injection(noise_type='gaussian', intensity=0.1, probability=0.4):
    """
    Add random noise to the image for data augmentation.
    
    Args:
        noise_type (str): Type of noise ('gaussian', 'uniform') (default: 'gaussian')
        intensity (float): Noise intensity (0.0 to 1.0) (default: 0.1)
        probability (float): Probability of applying noise (default: 0.4)
    """
    def wrapper(image, label):
        if tf.random.uniform(()) > probability:
            return image, label
            
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
        return noisy_image, label
    
    return wrapper


def tf_random_brightness(max_delta=0.2):
    """
    Apply random brightness adjustment to the image.
    
    Args:
        max_delta (float): Maximum brightness change (default: 0.2)
    """
    def wrapper(image, label):
        image = tf.image.stateless_random_brightness(
            image, 
            max_delta, 
            tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
        )
        return image, label
    
    return wrapper


def tf_random_contrast(lower=0.8, upper=1.2):
    """
    Apply random contrast adjustment to the image.
    
    Args:
        lower (float): Lower bound for contrast factor (default: 0.8)
        upper (float): Upper bound for contrast factor (default: 1.2)
    """
    def wrapper(image, label):
        image = tf.image.stateless_random_contrast(
            image,
            lower,
            upper,
            tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
        )
        return image, label
    
    return wrapper


def tf_random_saturation(lower=0.7, upper=1.3):
    """
    Apply random saturation adjustment to the image.
    
    Args:
        lower (float): Lower bound for saturation factor (default: 0.7)
        upper (float): Upper bound for saturation factor (default: 1.3)
    """
    def wrapper(image, label):
        image = tf.image.stateless_random_saturation(
            image,
            lower,
            upper,
            tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
        )
        return image, label
    
    return wrapper


def tf_random_hue(max_delta=0.1):
    """
    Apply random hue adjustment to the image.
    
    Args:
        max_delta (float): Maximum hue change (default: 0.1)
    """
    def wrapper(image, label):
        image = tf.image.stateless_random_hue(
            image,
            max_delta,
            tf.random.uniform([2], maxval=2**31, dtype=tf.int32)
        )
        return image, label
    
    return wrapper


def tf_random_rotation(max_angle=15.0):
    """
    Apply random rotation to the image.
    
    Args:
        max_angle (float): Maximum rotation angle in degrees (default: 15.0)
    """
    def wrapper(image, label):
        # Convert angle to radians
        angle_rad = tf.random.uniform((), -max_angle * np.pi / 180, max_angle * np.pi / 180)
        
        # Apply rotation using tf.keras.utils.image_utils
        rotated = tf.keras.utils.image_utils.random_rotation(
            tf.expand_dims(image, 0),
            angle_rad,
            row_axis=1,
            col_axis=2,
            channel_axis=3
        )[0]
        
        return rotated, label
    
    return wrapper

