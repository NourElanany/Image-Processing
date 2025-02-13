import numpy as np
from PIL import Image

def Negative(image):
    """
    Create negative of an image
    Args:
        image: Input image
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Invert the image
    negative = 255 - image
    
    return negative.astype(np.uint8)
