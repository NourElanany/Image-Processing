import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def WeightFilter(image, kernel=None):
    """
    Apply weighted filter to an image
    Args:
        image: Input image
        kernel: Custom weight kernel (default: Gaussian-like kernel)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Default kernel (Gaussian-like)
    if kernel is None:
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16.0
    
    # Handle RGB images
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for channel in range(image.shape[2]):
            filtered[:,:,channel] = convolve(image[:,:,channel], kernel)
    else:
        filtered = convolve(image, kernel)
    
    # Clip values and convert to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return filtered
