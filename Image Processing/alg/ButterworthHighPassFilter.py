import numpy as np
from scipy.fft import fft2, ifft2
from PIL import Image

def ButterworthHighPassFilter(image, cutoff=30, order=2):
    """
    Apply Butterworth High Pass Filter to an image
    Args:
        image: Input frequency domain data
        cutoff: Cutoff frequency (default: 30)
        order: Filter order (default: 2)
    Returns:
        numpy.ndarray: Filtered frequency domain data
    """
    if isinstance(image, Image.Image):
        raise ValueError("Expected frequency domain data, got PIL Image")
    
    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for filter
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distances
    D = np.sqrt(u**2 + v**2)
    
    # Create Butterworth high-pass filter
    H = 1 / (1 + (cutoff / (D + 1e-6))**(2 * order))
    
    # Apply filter to frequency domain data
    return image * H
