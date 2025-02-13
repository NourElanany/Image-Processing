import numpy as np
from PIL import Image

def GaussianNoise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image
    Args:
        image: Input image
        mean: Mean of the Gaussian distribution (default: 0)
        sigma: Standard deviation of the Gaussian distribution (default: 25)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
    
    # Generate noise with same shape as image
    noise = np.random.normal(mean * 255, sigma * 255, image.shape)
    
    # Add noise and clip values
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image
