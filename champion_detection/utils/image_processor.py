"""
Image processing utilities
"""

import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handle image loading and preprocessing"""
    
    def __init__(self):
        """Initialize image processor"""
        pass
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (RGB) or None if failed
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None
            
            # Load with OpenCV (BGR)
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image array
        
        Args:
            image: Image as numpy array
            
        Returns:
            True if valid, False otherwise
        """
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False
        
        if image.size == 0:
            return False
        
        return True
    
    def resize_image(self, image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
        """
        Resize image to specified size
        
        Args:
            image: Input image array
            size: Target size (width, height)
            
        Returns:
            Resized image array
        """
        return cv2.resize(image, size)