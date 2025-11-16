"""
YOLO Champion Card Detection
Pure object detection - only finds champion card bounding boxes
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLOCardDetector:
    """
    Pure YOLO object detection for champion cards
    
    Responsibilities:
    - Load trained YOLO model
    - Detect champion card bounding boxes
    - Return clean bounding box coordinates
    - No classification - just detection
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
        """
        logger.info("🎯 Initializing YOLO Card Detector")
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"✅ YOLO detector ready on {self.device}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Using default YOLOv8n model for now")
            return YOLO("yolov8n.pt")  # Default model
        
        logger.info(f"Loading YOLO model: {self.model_path}")
        return YOLO(str(self.model_path))
    
    def detect_cards(self, 
                    image: np.ndarray,
                    confidence_threshold: float = 0.5,
                    nms_threshold: float = 0.4) -> List[Dict]:
        """
        Detect champion cards in image
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            List of detections with bounding boxes and confidence scores
            Format: [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
        """
        
        try:
            # Run YOLO inference
            results = self.model(
                image,
                conf=confidence_threshold,
                iou=nms_threshold,
                verbose=False
            )
            
            detections = []
            
            # Parse results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        detection = {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                            "width": x2 - x1,
                            "height": y2 - y1
                        }
                        detections.append(detection)
            
            # Sort by confidence (highest first)
            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"🎯 Detected {len(detections)} champion cards")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def crop_detected_cards(self, 
                          image: np.ndarray, 
                          detections: List[Dict]) -> List[Dict]:
        """
        Crop champion cards from detected bounding boxes
        
        Args:
            image: Original image
            detections: List of detection results from detect_cards()
            
        Returns:
            List with cropped images added to each detection
            Format: [{"bbox": [...], "confidence": float, "crop": np.ndarray}, ...]
        """
        
        cropped_detections = []
        
        for i, detection in enumerate(detections):
            try:
                x1, y1, x2, y2 = detection["bbox"]
                
                # Validate bounds
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Crop the region
                crop = image[y1:y2, x1:x2]
                
                if crop.size > 0:
                    detection_with_crop = detection.copy()
                    detection_with_crop["crop"] = crop
                    detection_with_crop["crop_id"] = i
                    cropped_detections.append(detection_with_crop)
                else:
                    logger.warning(f"Empty crop for detection {i}")
                    
            except Exception as e:
                logger.error(f"Failed to crop detection {i}: {e}")
                continue
        
        logger.info(f"✂️ Successfully cropped {len(cropped_detections)} champion cards")
        return cropped_detections
    
    def visualize_detections(self, 
                           image: np.ndarray, 
                           detections: List[Dict],
                           show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image: Input image
            detections: Detection results
            show_confidence: Whether to show confidence scores
            
        Returns:
            Image with bounding boxes drawn
        """
        
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_confidence:
                label = f"Card {i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image
    
    def detect_and_crop(self, 
                       image: np.ndarray,
                       confidence_threshold: float = 0.5) -> List[Dict]:
        """
        One-step function: detect cards and return crops
        
        Args:
            image: Input image
            confidence_threshold: Detection confidence threshold
            
        Returns:
            List of detections with crops ready for CLIP classification
        """
        
        # Detect cards
        detections = self.detect_cards(image, confidence_threshold)
        
        if not detections:
            logger.warning("No champion cards detected")
            return []
        
        # Crop detected regions
        cropped_detections = self.crop_detected_cards(image, detections)
        
        return cropped_detections


def main():
    """Test YOLO detector"""
    
    # Initialize detector
    detector = YOLOCardDetector()
    
    # Test with a sample image (if available)
    test_image_path = "../test_images/banpick_sample.jpg"
    
    if Path(test_image_path).exists():
        # Load test image
        image = cv2.imread(test_image_path)
        
        # Detect and crop
        results = detector.detect_and_crop(image)
        
        print(f"Detected {len(results)} champion cards:")
        for i, result in enumerate(results):
            print(f"  Card {i+1}: confidence={result['confidence']:.3f}, "
                  f"bbox={result['bbox']}, crop_shape={result['crop'].shape}")
        
        # Visualize
        vis_image = detector.visualize_detections(image, results)
        cv2.imwrite("yolo_detection_result.jpg", vis_image)
        print("Visualization saved to: yolo_detection_result.jpg")
    else:
        print(f"Test image not found: {test_image_path}")
        print("YOLO detector initialized successfully!")


if __name__ == "__main__":
    main()