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
from typing import Union, Tuple


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


def power_law_transform(data: np.ndarray, 
                       gamma: float = 1.0,
                       scale_factor: float = 1.0) -> np.ndarray:
    """
    Apply power-law (gamma) transformation for brightness/contrast adjustment.
    
    The power-law transformation is useful for enhancing images that are 
    either too bright or too dark. Gamma < 1 brightens the image, 
    gamma > 1 darkens it.
    
    Args:
        data: Input image array
        gamma: Gamma value for power-law transformation
        scale_factor: Scaling factor applied before gamma correction
        
    Returns:
        Gamma-corrected image
    """
    # Normalize data to [0, 1] range
    normalized = data.astype(np.float32)
    if normalized.max() > 1.0:
        normalized = normalized / 255.0
    
    # Apply power-law transformation
    corrected = scale_factor * np.power(normalized, gamma)
    
    # Clip values to valid range
    corrected = np.clip(corrected, 0, 1)
    
    # Convert back to original data type
    if data.dtype == np.uint8:
        return (corrected * 255).astype(np.uint8)
    else:
        return corrected.astype(data.dtype)


def bilateral_filter_smooth(data: np.ndarray,
                           diameter: int = 9,
                           sigma_color: float = 75.0,
                           sigma_space: float = 75.0) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving smoothing.
    
    Bilateral filter reduces noise while preserving sharp edges, making it
    ideal for preprocessing images before feature extraction.
    
    Args:
        data: Input image array
        diameter: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Smoothed image with preserved edges
    """
    if len(data.shape) == 3:  # Color image
        return cv2.bilateralFilter(data.astype(np.uint8), 
                                 diameter, sigma_color, sigma_space)
    elif len(data.shape) == 2:  # Grayscale
        return cv2.bilateralFilter(data.astype(np.uint8), 
                                 diameter, sigma_color, sigma_space)
    else:
        return data


def morphological_operations(data: np.ndarray,
                           operation: str = "opening",
                           kernel_size: int = 5,
                           iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operations for shape-based image processing.
    
    Useful for removing noise, filling gaps, or enhancing structures
    in binary or grayscale images.
    
    Args:
        data: Input image array
        operation: Type of operation (opening, closing, erosion, dilation)
        kernel_size: Size of the morphological kernel
        iterations: Number of times to apply the operation
        
    Returns:
        Processed image after morphological operations
    """
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (kernel_size, kernel_size))
    
    # Map operation names to OpenCV constants
    operations = {
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE,
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT
    }
    
    if operation not in operations:
        operation = "opening"
    
    return cv2.morphologyEx(data.astype(np.uint8), 
                          operations[operation], 
                          kernel, iterations=iterations)


def frequency_domain_filter(data: np.ndarray,
                          filter_type: str = "low_pass",
                          cutoff_freq: float = 0.1,
                          order: int = 2) -> np.ndarray:
    """
    Apply frequency domain filtering (FFT-based).
    
    Useful for removing periodic noise or enhancing specific frequency
    components in the image.
    
    Args:
        data: Input image array
        filter_type: Type of filter (low_pass, high_pass, band_pass)
        cutoff_freq: Cutoff frequency (0-0.5, normalized)
        order: Filter order (higher = sharper cutoff)
        
    Returns:
        Filtered image in spatial domain
    """
    # Convert to grayscale if needed
    if len(data.shape) == 3:
        gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    else:
        gray = data.astype(np.float32)
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create frequency domain filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate arrays
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)
    
    # Distance from center
    D = np.sqrt(U**2 + V**2)
    D_norm = D / (min(rows, cols) / 2)  # Normalize
    
    # Create filter
    if filter_type == "low_pass":
        H = 1 / (1 + (D_norm / cutoff_freq)**(2 * order))
    elif filter_type == "high_pass":
        H = 1 / (1 + (cutoff_freq / (D_norm + 1e-6))**(2 * order))
    else:  # band_pass (simplified)
        low_cutoff = cutoff_freq * 0.5
        high_cutoff = cutoff_freq * 1.5
        H_low = 1 / (1 + (D_norm / high_cutoff)**(2 * order))
        H_high = 1 / (1 + (low_cutoff / (D_norm + 1e-6))**(2 * order))
        H = H_low - H_high
    
    # Apply filter
    filtered = f_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(f_ishift)
    result = np.abs(result)
    
    # Convert back to original format
    result = np.clip(result, 0, 255).astype(data.dtype)
    
    # If original was color, convert back
    if len(data.shape) == 3:
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    return result


def adaptive_threshold(data: np.ndarray,
                      max_value: float = 255.0,
                      adaptive_method: str = "gaussian",
                      block_size: int = 11,
                      constant: float = 2.0) -> np.ndarray:
    """
    Apply adaptive thresholding for binarization.
    
    Useful for creating binary masks or preparing images for segmentation
    when lighting conditions vary across the image.
    
    Args:
        data: Input image array (will be converted to grayscale)
        max_value: Maximum value assigned to pixels above threshold
        adaptive_method: Method for computing threshold (gaussian, mean)
        block_size: Size of neighborhood for threshold calculation
        constant: Constant subtracted from the weighted mean
        
    Returns:
        Binary image with adaptive thresholding applied
    """
    # Convert to grayscale if needed
    if len(data.shape) == 3:
        gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    else:
        gray = data.astype(np.uint8)
    
    # Select adaptive method
    if adaptive_method == "gaussian":
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:  # mean
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    
    # Apply adaptive threshold
    result = cv2.adaptiveThreshold(gray, max_value, method,
                                 cv2.THRESH_BINARY, block_size, constant)
    
    # Convert back to 3-channel if original was color
    if len(data.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return result
