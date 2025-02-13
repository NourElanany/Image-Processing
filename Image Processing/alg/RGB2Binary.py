import numpy as np
from PIL import Image

def RGB2Binary(image, threshold=127):
    """
    Convert RGB image to binary
    Args:
        image: Input RGB image
        threshold: Binarization threshold (default: 127)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
    
    # Convert to grayscale first if RGB
    if len(image.shape) == 3:
        # Preserve dimensions for proper display
        grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        # Add back the channel dimension
        grayscale = np.stack([grayscale] * 3, axis=-1)
    else:
        grayscale = image
    
    # Convert to binary
    binary = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
    
    return binary
