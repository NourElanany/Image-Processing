import numpy as np
from PIL import Image

def GammaCorrection(image, gamma=1.0):
    """
    Apply gamma correction to an image
    Args:
        image: Input image
        gamma: Gamma value (default: 1.0)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Normalize to 0-1 range
    normalized = image.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Convert back to 0-255 range
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(corrected)
