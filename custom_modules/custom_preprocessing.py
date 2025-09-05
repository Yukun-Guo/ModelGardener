import numpy as np
import cv2
import tensorflow as tf

def adaptive_histogram_equalization(clip_limit=2.0, tile_grid_size=8):
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

