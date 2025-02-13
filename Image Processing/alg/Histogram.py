import numpy as np
from PIL import Image

def Histogram(image):
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
        # Apply histogram equalization to each channel
        equalized = np.zeros_like(image)
        for channel in range(image.shape[2]):
            equalized[:,:,channel] = equalize_channel(image[:,:,channel])
    else:
        # Apply histogram equalization to grayscale image
        equalized = equalize_channel(image)
    
    return equalized.astype(np.uint8)

def equalize_channel(channel):
    """Helper function to equalize a single channel"""
    # Calculate histogram
    hist, bins = np.histogram(channel.flatten(), bins=256, range=(0,256))
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Use linear interpolation to map input values to equalized values
    equalized = np.interp(channel.flatten(), bins[:-1], cdf)
    
    # Reshape back to original shape
    return equalized.reshape(channel.shape)
