import numpy as np
import cv2
import tensorflow as tf

def color_shift(hue_shift=20, saturation_scale=1.2, value_scale=1.1, probability=0.6):
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

