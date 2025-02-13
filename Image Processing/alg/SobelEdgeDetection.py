import numpy as np
from PIL import Image

def SobelEdgeDetection(image, threshold=30):
    """
    Apply Sobel edge detection to an image
    Args:
        image: Input image
        threshold: Edge detection threshold (default: 30)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Sobel operators
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    rows, cols = image.shape
    edges = np.zeros_like(image)
    
    # Apply Sobel operators
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = np.sum(image[i-1:i+2, j-1:j+2] * Gx)
            gy = np.sum(image[i-1:i+2, j-1:j+2] * Gy)
            magnitude = np.sqrt(gx**2 + gy**2)
            edges[i, j] = 255 if magnitude > threshold else 0
    
    return Image.fromarray(edges.astype(np.uint8))
