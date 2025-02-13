import numpy as np
from PIL import Image

def Brightness(image, factor=1.0):
    """
    Adjust the brightness of an image
    Args:
        image: Input image
        factor: Brightness factor (default: 1.0)
               Values > 1.0 increase brightness
               Values < 1.0 decrease brightness
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply brightness adjustment
    brightened = image * factor
    
    # Clip values to valid range
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    return brightened