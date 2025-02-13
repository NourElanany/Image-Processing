import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

def MidPointFilter(image, size=3):
    """
    Apply midpoint filter to an image
    Args:
        image: Input image
        size: Filter window size
    """
    # Convert to float for calculations
    img_float = np.array(image, dtype=np.float32)
    
    # Calculate local min and max using uniform filter
    local_min = uniform_filter(np.minimum.reduce(img_float, axis=2 if len(img_float.shape) > 2 else None), size)
    local_max = uniform_filter(np.maximum.reduce(img_float, axis=2 if len(img_float.shape) > 2 else None), size)
    
    # Calculate midpoint
    result = (local_min + local_max) / 2
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # If input was RGB, broadcast result to all channels
    if len(image.shape) > 2:
        result = np.stack([result] * 3, axis=-1)
    
    return result