import numpy as np
from PIL import Image

def GammaNoise(image, shape=1.0, scale=1.0):
    """
    Add gamma noise to an image
    Args:
        image: Input image
        shape: Shape parameter for gamma distribution
        scale: Scale parameter for gamma distribution
    """
    noise = np.random.gamma(shape, scale, size=image.shape)
    noisy_image = np.array(image, dtype=np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
