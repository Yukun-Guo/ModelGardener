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


def random_pixelate(image, block_size=8, probability=0.5):
    """
    Apply random pixelation effect to image.
    
    Args:
        image (np.ndarray): Input image
        block_size (int): Size of pixelation blocks (default: 8)
        probability (float): Probability of applying effect (default: 0.5)
    
    Returns:
        np.ndarray: Pixelated image
    """
    if np.random.random() > probability:
        return image
    
    try:
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Resize down and then back up to create pixelation effect
        temp = cv2.resize(image, 
                         (width // block_size, height // block_size), 
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, 
                              (width, height), 
                              interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    except Exception as e:
        print(f"Error in random_pixelate: {e}")
        return image


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


def add_rain_effect(image, rain_density=50, rain_length=20, rain_angle=0, probability=0.3):
    """
    Add simulated rain effect to image.
    
    Args:
        image (np.ndarray): Input image
        rain_density (int): Number of rain drops (default: 50)
        rain_length (int): Length of rain streaks (default: 20)
        rain_angle (int): Angle of rain in degrees (default: 0)
        probability (float): Probability of applying effect (default: 0.3)
    
    Returns:
        np.ndarray: Image with rain effect
    """
    if np.random.random() > probability:
        return image
    
    try:
        result = image.copy()
        height, width = result.shape[:2]
        
        # Generate random rain drops
        for _ in range(rain_density):
            # Random position
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # Calculate end point based on angle and length
            end_x = int(x + rain_length * np.cos(np.radians(rain_angle)))
            end_y = int(y + rain_length * np.sin(np.radians(rain_angle)))
            
            # Ensure end point is within image bounds
            end_x = max(0, min(width - 1, end_x))
            end_y = max(0, min(height - 1, end_y))
            
            # Draw rain streak
            cv2.line(result, (x, y), (end_x, end_y), (200, 200, 255), 1)
        
        return result
        
    except Exception as e:
        print(f"Error in add_rain_effect: {e}")
        return image


def lens_distortion(image, distortion_strength=0.3, probability=0.4):
    """
    Apply lens distortion effect (barrel or pincushion).
    
    Args:
        image (np.ndarray): Input image
        distortion_strength (float): Strength of distortion effect (default: 0.3)
        probability (float): Probability of applying effect (default: 0.4)
    
    Returns:
        np.ndarray: Distorted image
    """
    if np.random.random() > probability:
        return image
    
    try:
        height, width = image.shape[:2]
        
        # Camera matrix (simplified)
        camera_matrix = np.array([[width, 0, width/2],
                                 [0, height, height/2],
                                 [0, 0, 1]], dtype=np.float32)
        
        # Distortion coefficients (k1, k2, p1, p2, k3)
        # Positive k1 creates barrel distortion, negative creates pincushion
        k1 = np.random.uniform(-distortion_strength, distortion_strength)
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
        
        # Apply distortion
        result = cv2.undistort(image, camera_matrix, dist_coeffs)
        return result
        
    except Exception as e:
        print(f"Error in lens_distortion: {e}")
        return image


def vintage_filter(image, sepia_strength=0.8, vignette_strength=0.5, probability=0.25):
    """
    Apply vintage/retro filter effect.
    
    Args:
        image (np.ndarray): Input image
        sepia_strength (float): Strength of sepia effect (default: 0.8)
        vignette_strength (float): Strength of vignette effect (default: 0.5)
        probability (float): Probability of applying effect (default: 0.25)
    
    Returns:
        np.ndarray: Vintage-filtered image
    """
    if np.random.random() > probability:
        return image
    
    try:
        result = image.copy().astype(np.float32)
        height, width = result.shape[:2]
        
        # Apply sepia effect
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        
        sepia_img = cv2.transform(result, sepia_kernel)
        result = result * (1 - sepia_strength) + sepia_img * sepia_strength
        
        # Add vignette effect
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1 - (distances / max_distance) * vignette_strength
        vignette = np.clip(vignette, 0, 1)
        
        # Apply vignette to all channels
        for i in range(result.shape[2]):
            result[:, :, i] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Error in vintage_filter: {e}")
        return image
