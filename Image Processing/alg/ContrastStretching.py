import numpy as np
from PIL import Image

def ContrastStretching(image, low_percentile=2, high_percentile=98):
    """
    Apply contrast stretching to an image
    Args:
        image: Input image
        low_percentile: Lower percentile for stretching (default: 2)
        high_percentile: Upper percentile for stretching (default: 98)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle RGB images
    if len(image.shape) == 3:
        stretched = np.zeros_like(image)
        for channel in range(image.shape[2]):
            stretched[:,:,channel] = stretch_channel(image[:,:,channel], 
                                                  low_percentile, 
                                                  high_percentile)
    else:
        stretched = stretch_channel(image, low_percentile, high_percentile)
    
    return stretched.astype(np.uint8)

def stretch_channel(channel, low_percentile, high_percentile):
    """Helper function to stretch a single channel"""
    # Calculate percentiles
    low = np.percentile(channel, low_percentile)
    high = np.percentile(channel, high_percentile)
    
    # Stretch the channel
    stretched = np.clip((channel - low) * 255.0 / (high - low), 0, 255)
    
    return stretched
