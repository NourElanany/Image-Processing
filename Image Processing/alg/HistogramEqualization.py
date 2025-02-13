import numpy as np
from PIL import Image

def HistogramEqualization(image):
    """
    Apply histogram equalization to an image
    Args:
        image: Input image
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle RGB images
    if len(image.shape) == 3:
        # Convert to HSV for better equalization
        hsv = np.array(Image.fromarray(image).convert('HSV'))
        # Apply equalization to V channel
        hsv[:,:,2] = equalize_channel(hsv[:,:,2])
        # Convert back to RGB
        equalized = np.array(Image.fromarray(hsv, mode='HSV').convert('RGB'))
    else:
        equalized = equalize_channel(image)
    
    return equalized.astype(np.uint8)

def equalize_channel(channel):
    """Helper function to equalize a single channel"""
    # Calculate histogram
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # Create lookup table
    lookup_table = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
    
    # Reshape back to original shape
    return lookup_table.reshape(channel.shape)
