import numpy as np
from PIL import Image

def RayleighNoise(image, scale=0.1):
    """
    Add Rayleigh noise to an image
    Args:
        image: Input image
        scale: Scale parameter for Rayleigh distribution (default: 0.1)
              Higher values create more intense noise
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
    
    # Generate Rayleigh noise
    noise = np.random.rayleigh(scale * 255, image.shape)
    
    # Add noise and clip values
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image