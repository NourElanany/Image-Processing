import numpy as np
from PIL import Image

def ExponentialNoise(image, scale=1.0):
    """
    Add exponential noise to an image
    Args:
        image: Input image
        scale: Scale parameter for exponential distribution
    """
    noise = np.random.exponential(scale=scale, size=image.shape)
    noisy_image = np.array(image, dtype=np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)