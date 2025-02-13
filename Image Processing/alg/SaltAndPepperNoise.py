import numpy as np
from PIL import Image

def SaltAndPepperNoise(image, prob=0.05):
    """
    Add salt and pepper noise to an image
    Args:
        image: Input image
        prob: Probability of noise (default: 0.05)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
    
    noisy_image = np.copy(image)
    
    # Generate single mask for all channels if RGB
    if len(image.shape) == 3:
        # Salt noise
        salt_mask = np.random.random(image.shape[:2]) < (prob/2)
        salt_mask = np.dstack([salt_mask] * 3)
        noisy_image[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = np.random.random(image.shape[:2]) < (prob/2)
        pepper_mask = np.dstack([pepper_mask] * 3)
        noisy_image[pepper_mask] = 0
    else:
        # Salt noise for grayscale
        salt_mask = np.random.random(image.shape) < (prob/2)
        noisy_image[salt_mask] = 255
        
        # Pepper noise for grayscale
        pepper_mask = np.random.random(image.shape) < (prob/2)
        noisy_image[pepper_mask] = 0
    
    return noisy_image.astype(np.uint8)
