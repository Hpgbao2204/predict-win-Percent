"""
Utility functions for image processing and data handling
"""

import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from config.settings import *

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handle image loading, preprocessing, and validation"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of image or None if failed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
                
            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                logger.error(f"Unsupported image format: {image_path.suffix}")
                return None
                
            # Load with PIL first (better format support)
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = MAX_IMAGE_SIZE) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize while maintaining aspect ratio
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas and center the image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if image is suitable for processing
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image is valid
        """
        if image is None:
            return False
            
        # Check dimensions
        if len(image.shape) != 3:
            return False
            
        h, w, c = image.shape
        if h < 32 or w < 32 or c != 3:
            return False
            
        return True

class DataLoader:
    """Handle loading and managing champion data"""
    
    def __init__(self):
        self.champions_data = None
        self.simple_mapping = None
        self._load_data()
    
    def _load_data(self):
        """Load champion data from JSON files"""
        try:
            # Load detailed champion data
            if SKINS_DETAILED_JSON.exists():
                with open(SKINS_DETAILED_JSON, 'r', encoding='utf-8') as f:
                    self.champions_data = json.load(f)
                logger.info(f"Loaded detailed data for {len(self.champions_data)} champions")
            
            # Load simple mapping
            if SKINS_SIMPLE_JSON.exists():
                with open(SKINS_SIMPLE_JSON, 'r', encoding='utf-8') as f:
                    self.simple_mapping = json.load(f)
                logger.info(f"Loaded simple mapping for {len(self.simple_mapping)} champions")
                
        except Exception as e:
            logger.error(f"Error loading champion data: {str(e)}")
    
    def get_champion_names(self) -> List[str]:
        """Get list of all champion names"""
        if self.champions_data:
            return [data["original_name"] for data in self.champions_data.values()]
        return []
    
    def get_champion_skins(self, champion_key: str) -> List[str]:
        """Get all skins for a specific champion"""
        if self.champions_data and champion_key in self.champions_data:
            return self.champions_data[champion_key]["skins"]
        return []
    
    def get_default_skin(self, champion_key: str) -> Optional[str]:
        """Get default skin path for a champion"""
        if self.simple_mapping and champion_key in self.simple_mapping:
            return self.simple_mapping[champion_key]
        return None
    
    def get_champion_info(self, champion_key: str) -> Optional[Dict]:
        """Get complete champion information"""
        if self.champions_data and champion_key in self.champions_data:
            return self.champions_data[champion_key]
        return None

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / "champion_detection.log")
        ]
    )

def create_results_directory():
    """Create results directory if it doesn't exist"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)