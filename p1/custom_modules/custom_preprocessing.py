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

