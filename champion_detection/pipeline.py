#!/usr/bin/env python3
"""
YOLO + CLIP Champion Detection Pipeline
Clean, simple pipeline for champion detection
"""

import cv2
import numpy as np
from typing import List, Dict
import logging
from pathlib import Path

from models.yolo.card_detector import YOLOCardDetector
from models.clip.champion_classifier import CLIPChampionClassifier

logger = logging.getLogger(__name__)

class ChampionDetectionPipeline:
    """Simple YOLO + CLIP pipeline"""
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        """Initialize the pipeline"""
        print("🚀 Initializing Champion Detection Pipeline...")
        
        # Initialize components
        self.yolo_detector = YOLOCardDetector(yolo_model_path)
        self.clip_classifier = CLIPChampionClassifier()
        
        print("✅ Pipeline ready!")
    
    def detect_champions(self, image: np.ndarray) -> List[Dict]:
        """Main detection function"""
        
        # Step 1: YOLO detects card positions
        detections = self.yolo_detector.detect_and_crop(image)
        
        if not detections:
            return []
        
        # Step 2: CLIP classifies champions
        results = self.clip_classifier.classify_detections(detections)
        
        return results

def main():
    """Test the pipeline"""
    pipeline = ChampionDetectionPipeline()
    print("Pipeline initialized successfully!")

if __name__ == "__main__":
    main()