import numpy as np
from scipy.fft import fft2, ifft2
from PIL import Image

def GaussianHighPassFilter(image, sigma=30):
    """
    Apply Gaussian High Pass Filter to an image
    Args:
        image: Input image
        sigma: Standard deviation of Gaussian filter (default: 30)
    """
    # Convert to grayscale if needed
    if isinstance(image, Image.Image):
        image = image.convert('L')
        image = np.array(image)
    elif isinstance(image, np.ndarray) and len(image.shape) == 3:
        # Convert RGB to grayscale using weighted sum
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for filter
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distances
    D = np.sqrt(u**2 + v**2)
    
    # Create Gaussian high-pass filter
    H = 1 - np.exp(-(D**2) / (2 * sigma**2))
    
    # Apply filter in frequency domain
    F = fft2(image)
    F = np.fft.fftshift(F)
    G = F * H
    G = np.fft.ifftshift(G)
    
    # Inverse transform
    filtered_image = np.real(ifft2(G))
    
    # Normalize and convert to uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

def GaussianHighPassFilterFreq(image, sigma=1.0):
    """
    Apply Gaussian High Pass Filter to frequency domain data
    Args:
        image: Input frequency domain data
        sigma: Standard deviation (default: 1.0)
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
    
    # Create Gaussian high-pass filter
    H = 1 - np.exp(-(D**2) / (2 * sigma**2))
    
    # Apply filter to frequency domain data
    return image * H
