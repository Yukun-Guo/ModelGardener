"""
Example custom preprocessing functions for the Model Gardener application.

All functions follow the nested wrapper pattern where:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (data, label) and returns (processed_data, label)
- Configuration parameters are set at the outer function level

Example usage pattern:
def preprocessing_name(param1=default1, param2=default2):
    def wrapper(data, label):
        # Apply preprocessing logic here
        processed_data = apply_preprocessing(data, param1, param2)
        return processed_data, label
    return wrapper

The function signature should be: def function_name(param1=default1, param2=default2, ...):
"""

import numpy as np
import cv2
import tensorflow as tf

def tf_adaptive_histogram_equalization(clip_limit=2.0, tile_grid_size=8):
    """
    Apply adaptive histogram equalization (CLAHE) to improve image contrast.
    
    This method enhances local contrast in images by applying histogram 
    equalization to small regions (tiles) rather than the entire image.
    
    Args:
        clip_limit (float): Threshold for contrast limiting (higher = more contrast) (default: 2.0)
        tile_grid_size (int): Size of the grid for adaptive equalization (default: 8)
    """
    def wrapper(data, label):
        # Convert TensorFlow tensor to numpy for OpenCV operations
        if tf.is_tensor(data):
            np_data = data.numpy()
        else:
            np_data = data
            
        if len(np_data.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=(tile_grid_size, tile_grid_size))
            processed = clahe.apply(np_data.astype(np.uint8))
        elif len(np_data.shape) == 3:  # Color image
            # Convert to LAB color space for better results
            if np_data.max() <= 1.0:  # Normalized image
                np_data = (np_data * 255).astype(np.uint8)
            lab = cv2.cvtColor(np_data.astype(np.uint8), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=(tile_grid_size, tile_grid_size))
            lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply only to L channel
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            if data.dtype == np.float32 or data.dtype == np.float64:
                processed = processed.astype(np.float32) / 255.0
        else:
            processed = np_data
            
        # Convert back to TensorFlow tensor if input was tensor
        if tf.is_tensor(data):
            processed = tf.constant(processed, dtype=data.dtype)
            
        return processed, label
    
    return wrapper


def tf_edge_enhancement(strength=1.5, blur_radius=3):
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


def tf_gamma_correction(gamma=1.2, gain=1.0):
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


def tf_normalize_custom(mean_values=[0.485, 0.456, 0.406], std_values=[0.229, 0.224, 0.225]):
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


def tf_resize_with_pad(target_height=224, target_width=224, pad_value=0):
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
