import numpy as np
from PIL import Image

def Rgb2Gray(image):
    """
    Convert RGB image to grayscale
    Args:
        image: Input RGB image
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Check if image is already grayscale
    if len(image.shape) == 2:
        return image
    
    # Convert to grayscale using weighted sum
    grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    return grayscale.astype(np.uint8)
