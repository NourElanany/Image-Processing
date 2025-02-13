import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift

def FourierTransform(image):
    """
    Apply Fourier Transform to an image and return the magnitude spectrum and phase
    Args:
        image: Input image (PIL Image or numpy array)
    Returns:
        tuple: (magnitude spectrum for display, complex frequency domain for inverse transform)
    """
    # Convert to grayscale if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image)
    else:
        img_array = image
        if len(img_array.shape) > 2:
            # Convert RGB to grayscale using standard weights
            img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    img_array = img_array.astype(np.float32)
    
    # Apply FFT and shift
    f_transform = fft2(img_array)
    f_shift = fftshift(f_transform)
    
    # Calculate magnitude spectrum for display
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Normalize to 0-255 range for display
    display_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) * 255
                       / (magnitude_spectrum.max() - magnitude_spectrum.min()))
    
    # Return both display spectrum and the complex frequency domain
    return display_spectrum.astype(np.uint8), f_shift