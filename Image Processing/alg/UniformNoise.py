import numpy as np
from PIL import Image

def UniformNoise(image, low=-0.2, high=0.2):
    """
    Add uniform noise to an image
    Args:
        image: Input image
        low: Lower bound of uniform distribution (default: -0.2)
        high: Upper bound of uniform distribution (default: 0.2)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
    
    # Generate uniform noise
    noise = np.random.uniform(low * 255, high * 255, image.shape)
    
    # Add noise and clip values
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image
