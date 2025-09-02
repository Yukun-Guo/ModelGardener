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


def random_blur(image, max_kernel_size=5, probability=0.5):
    """
    Apply random Gaussian blur to the image.
    
    Args:
        image: Input image (numpy array)
        max_kernel_size: Maximum blur kernel size (odd number)
        probability: Probability of applying blur
        
    Returns:
        Blurred or original image
    """
    try:
        if np.random.random() > probability:
            return image
            
        # Ensure kernel size is odd
        kernel_size = np.random.randint(1, max_kernel_size//2 + 1) * 2 + 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
        
    except Exception as e:
        print(f"Error in random_blur: {e}")
        return image


def noise_injection(image, noise_type='gaussian', intensity=0.1, probability=0.4):
    """
    Add random noise to the image for data augmentation.
    
    Args:
        image: Input image (numpy array)
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'uniform')
        intensity: Noise intensity (0.0 to 1.0)
        probability: Probability of applying noise
        
    Returns:
        Noisy or original image
    """
    try:
        if np.random.random() > probability:
            return image
            
        noisy_image = image.astype(np.float32)
        
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy_image += noise
        elif noise_type == 'uniform':
            # Uniform noise
            noise = np.random.uniform(-intensity * 255, intensity * 255, image.shape)
            noisy_image += noise
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            coords = [np.random.randint(0, i - 1, int(intensity * image.size * 0.1)) 
                     for i in image.shape[:2]]
            noisy_image[coords] = 255  # Salt
            
            coords = [np.random.randint(0, i - 1, int(intensity * image.size * 0.1)) 
                     for i in image.shape[:2]]
            noisy_image[coords] = 0   # Pepper
        
        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(image.dtype)
        
    except Exception as e:
        print(f"Error in noise_injection: {e}")
        return image

