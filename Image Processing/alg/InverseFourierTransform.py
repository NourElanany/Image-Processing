import numpy as np
from scipy.fft import ifft2, ifftshift
from PIL import Image

def InverseFourierTransform(freq_domain):
    """
    Apply Inverse Fourier Transform to a frequency domain image
    Args:
        freq_domain: Input frequency domain data (complex array from FourierTransform)
    Returns:
        numpy.ndarray: Reconstructed spatial domain image
    """
    # No need to convert PIL Image since we expect complex array
    if isinstance(freq_domain, Image.Image):
        raise ValueError("Expected complex frequency domain array, got PIL Image")
    
    # Apply inverse FFT
    f_ishift = ifftshift(freq_domain)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back).real  # Take real part and magnitude
    
    # Normalize to original range
    img_back = np.clip(img_back, 0, 255)
    
    return img_back.astype(np.uint8)
