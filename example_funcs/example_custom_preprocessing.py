"""
Example custom preprocessing functions for the Model Gardener application.

These functions demonstrate how to create custom preprocessing methods that can be 
dynamically loaded into the preprocessing parameter tree. Each function should:

1. Accept 'data' (or similar) as the first parameter (the input data)
2. Return the processed data
3. Have additional parameters for configuration
4. Include a docstring describing the function

The function signature should be: def function_name(data, param1=default1, param2=default2, ...):
"""

import numpy as np
import cv2

def adaptive_histogram_equalization(data: np.ndarray, 
                                   clip_limit: float = 2.0, 
                                   tile_grid_size: int = 8) -> np.ndarray:
    """
    Apply adaptive histogram equalization (CLAHE) to improve image contrast.
    
    This method enhances local contrast in images by applying histogram 
    equalization to small regions (tiles) rather than the entire image.
    
    Args:
        data: Input image array (grayscale or color)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of the grid for adaptive equalization
        
    Returns:
        Enhanced image with improved local contrast
    """
    if len(data.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                               tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(data.astype(np.uint8))
    
    elif len(data.shape) == 3:  # Color image
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                               tileGridSize=(tile_grid_size, tile_grid_size))
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply only to L channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return data


def edge_enhancement(data: np.ndarray, 
                    strength: float = 1.5, 
                    blur_radius: int = 3) -> np.ndarray:
    """
    Enhance edges in images using unsharp masking technique.
    
    Args:
        data: Input image array
        strength: Edge enhancement strength (higher = more enhancement)
        blur_radius: Gaussian blur radius for unsharp mask
        
    Returns:
        Edge-enhanced image
    """
    if len(data.shape) == 2:  # Grayscale
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(data.astype(np.float32), (blur_radius*2+1, blur_radius*2+1), 0)
        # Create unsharp mask
        mask = data.astype(np.float32) - blurred
        # Apply enhancement
        enhanced = data.astype(np.float32) + strength * mask
        return np.clip(enhanced, 0, 255).astype(data.dtype)
    
    elif len(data.shape) == 3:  # Color image
        enhanced = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[2]):
            channel = data[:, :, i].astype(np.float32)
            blurred = cv2.GaussianBlur(channel, (blur_radius*2+1, blur_radius*2+1), 0)
            mask = channel - blurred
            enhanced[:, :, i] = channel + strength * mask
        return np.clip(enhanced, 0, 255).astype(data.dtype)
    
    return data


def gamma_correction(data: np.ndarray, 
                    gamma: float = 1.2,
                    gain: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to adjust image brightness and contrast.
    
    Args:
        data: Input image array (0-255 or 0-1 range)
        gamma: Gamma value (>1 darkens, <1 brightens)
        gain: Gain factor applied before gamma correction
        
    Returns:
        Gamma-corrected image
    """
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
        return (corrected * 255).astype(data.dtype)
    else:
        return corrected.astype(data.dtype)
