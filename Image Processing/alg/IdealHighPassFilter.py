import numpy as np
from PIL import Image

def IdealHighPassFilter(image, cutoff=30):
    """
    Apply Ideal High Pass Filter to frequency domain data
    Args:
        image: Input frequency domain data
        cutoff: Cutoff frequency (default: 30)
    Returns:
        numpy.ndarray: Filtered frequency domain data
    """
    if isinstance(image, Image.Image):
        raise ValueError("Expected frequency domain data, got PIL Image")
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for filter
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distances
    D = np.sqrt(u**2 + v**2)
    
    # Create ideal high-pass filter
    H = (D > cutoff).astype(float)
    
    # Apply filter to frequency domain data
    return image * H
