import numpy as np
from PIL import Image


def PointSharpening(image, factor=1.5):
    """
    Apply point sharpening to an image
    Args:
        image: Input image
        factor: Sharpening factor (default: 1.5)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]]) * factor
    
    # Apply kernel to each channel if RGB
    if len(image.shape) == 3:
        sharpened = np.zeros_like(image)
        for channel in range(image.shape[2]):
            sharpened[:,:,channel] = convolve2d(image[:,:,channel], kernel)
    else:
        sharpened = convolve2d(image, kernel)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def convolve2d(image, kernel):
    """Helper function for 2D convolution"""
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape
    pad_rows = k_rows // 2
    pad_cols = k_cols // 2
    
    # Pad image
    padded = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='edge')
    
    # Apply convolution
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            result[i,j] = np.sum(padded[i:i+k_rows, j:j+k_cols] * kernel)
            
    return result