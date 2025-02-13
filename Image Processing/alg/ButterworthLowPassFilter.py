import numpy as np
from PIL import Image

def ButterworthLowPassFilter(image, cutoff=30, order=2):
    """
    Apply Butterworth Low Pass Filter to frequency domain data
    Args:
        image: Input frequency domain data
        cutoff: Cutoff frequency (default: 30)
        order: Filter order (default: 2)
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
    
    # Create Butterworth low-pass filter
    H = 1 / (1 + (D / cutoff)**(2 * order))
    
    # Apply filter to frequency domain data
    return image * H