"""
Example custom augmentation functions for ModelGardener.

These functions should accept an 'image' parameter as the first argument
and return a modified image. Additional parameters can be added for customization.

Requirements:
- Function should accept 'image' (numpy array) as first parameter
- Function should return modified image (numpy array) 
- Additional parameters are optional and will be extracted automatically
- Functions should handle errors gracefully
"""

import numpy as np
import cv2

def color_shift(image, hue_shift=20, saturation_scale=1.2, value_scale=1.1, probability=0.6):
    """
    Apply color shifting in HSV space.
    
    Args:
        image (np.ndarray): Input image
        hue_shift (int): Maximum hue shift in degrees (default: 20)
        saturation_scale (float): Saturation scaling factor (default: 1.2)
        value_scale (float): Value/brightness scaling factor (default: 1.1)
        probability (float): Probability of applying effect (default: 0.6)
    
    Returns:
        np.ndarray: Color-shifted image
    """
    if np.random.random() > probability:
        return image
    
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply random hue shift
        hue_delta = np.random.randint(-hue_shift, hue_shift + 1)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        
        # Apply saturation scaling
        sat_factor = np.random.uniform(1.0, saturation_scale)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Apply value scaling
        val_factor = np.random.uniform(1.0, value_scale)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result
        
    except Exception as e:
        print(f"Error in color_shift: {e}")
        return image

