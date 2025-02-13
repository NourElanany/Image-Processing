import numpy as np
from PIL import Image

def Gray2Binary(image, threshold=127):
    """
    Convert grayscale image to binary
    Args:
        image: Input grayscale image
        threshold: Binarization threshold (default: 127)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Convert to binary
    binary = (image > threshold).astype(np.uint8) * 255
    
    return binary
