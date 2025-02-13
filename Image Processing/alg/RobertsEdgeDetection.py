import numpy as np
from PIL import Image

def RobertsEdgeDetection(image, threshold=30):
    """
    Apply Roberts edge detection to an image
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
    
    # Convert to float and normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0
    
    # Roberts operators
    Gx = np.array([[1, 0],
                   [0, -1]])
    
    Gy = np.array([[0, 1],
                   [-1, 0]])
    
    rows, cols = image.shape
    edges = np.zeros_like(image)
    
    # Apply Roberts operators
    for i in range(rows - 1):
        for j in range(cols - 1):
            gx = np.sum(image[i:i+2, j:j+2] * Gx)
            gy = np.sum(image[i:i+2, j:j+2] * Gy)
            magnitude = np.sqrt(gx**2 + gy**2)
            # Normalize threshold to [0, 1] range
            norm_threshold = threshold / 100.0
            edges[i, j] = 1.0 if magnitude > norm_threshold else 0.0
    
    # Convert back to [0, 255] range
    edges = (edges * 255).astype(np.uint8)
    return Image.fromarray(edges)
