#!/usr/bin/env python3
"""
Ban-Pick Champion Detection System

Detects champion names from League of Legends ban-pick phase screenshots.

Input: Single ban-pick screenshot image
Output: List of detected champion names (up to 10)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import json
import sys

# Add champion_detection to path
sys.path.insert(0, str(Path(__file__).parent / "champion_detection"))

from pipeline import ChampionDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BanPickDetector:
    """
    Detection system for identifying champions in ban-pick screenshots.
    Uses YOLO for card detection and CLIP for champion classification.
    """
    
    def __init__(self, yolo_model_path: str = None):
        """
        Initialize the detector.
        
        Args:
            yolo_model_path: Path to trained YOLO model weights.
                           If None, searches for model in default locations.
        """
        if yolo_model_path is None:
            # Search for trained YOLO model
            model_paths = [
                "runs/detect/champion_card_detection/weights/best.pt",
                "champion_detection/yolov8n.pt",
                "yolov8n.pt"
            ]
            
            for path in model_paths:
                if Path(path).exists():
                    yolo_model_path = path
                    logger.info(f"Found YOLO model: {path}")
                    break
            
            if yolo_model_path is None:
                logger.warning("No trained YOLO model found, using default YOLOv8n")
                yolo_model_path = "yolov8n.pt"
        
        logger.info("Initializing Ban-Pick Detection System...")
        self.pipeline = ChampionDetectionPipeline(yolo_model_path)
        logger.info("System ready")
    
    def detect_from_file(self, image_path: str, min_champions: int = 10) -> List[Dict]:
        """
        Detect champions from image file.
        
        Args:
            image_path: Path to ban-pick screenshot
            min_champions: Minimum expected champions (default: 10)
            
        Returns:
            List of detection dictionaries containing:
            - champion_name: Champion name
            - confidence: Detection confidence score
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - position: Position metadata (team, phase, coordinates)
        """
        logger.info(f"Reading image: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        logger.info(f"Detecting champions in image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect champions
        results = self.pipeline.detect_champions(image)
        
        logger.info(f"Detected {len(results)} champions")
        
        # Classify positions
        results = self._classify_positions(results, image.shape)
        
        # Warning if insufficient detections
        if len(results) < min_champions:
            logger.warning(f"Only detected {len(results)}/{min_champions} champions")
        
        return results
    
    def _classify_positions(self, detections: List[Dict], image_shape: Tuple) -> List[Dict]:
        """
        Classify champion positions (blue/red team, ban/pick phase).
        Based on spatial location in the image.
        
        Args:
            detections: List of detections from pipeline
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Detections with added position metadata
        """
        height, width = image_shape[:2]
        
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            
            # Classify team based on horizontal position
            if x_center < width / 2:
                team = "blue_team"
            else:
                team = "red_team"
            
            # Classify phase based on vertical position
            # Typically bans are at top, picks below
            if y_center < height / 3:
                phase = "ban"
            else:
                phase = "pick"
            
            det['position'] = {
                'team': team,
                'phase': phase,
                'x_center': x_center,
                'y_center': y_center
            }
        
        return detections
    
    def get_champion_names(self, detections: List[Dict], top_n: int = 10) -> List[str]:
        """
        Extract champion names from detections.
        
        Args:
            detections: List of detections
            top_n: Number of champions to return (default: 10)
            
        Returns:
            List of champion names, sorted by confidence
        """
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Extract top N names
        champion_names = [det['champion_name'] for det in sorted_dets[:top_n]]
        
        return champion_names
    
    def print_results(self, detections: List[Dict]):
        """
        Print detection results to console.
        """
        print("\n" + "="*60)
        print("BAN-PICK DETECTION RESULTS")
        print("="*60)
        
        if not detections:
            print("No champions detected")
            return
        
        # Group by team
        blue_bans = []
        blue_picks = []
        red_bans = []
        red_picks = []
        
        for det in detections:
            pos = det.get('position', {})
            team = pos.get('team', 'unknown')
            phase = pos.get('phase', 'unknown')
            
            name = det['champion_name']
            conf = det.get('confidence', 0)
            
            entry = f"{name} ({conf:.2f})"
            
            if team == 'blue_team' and phase == 'ban':
                blue_bans.append(entry)
            elif team == 'blue_team' and phase == 'pick':
                blue_picks.append(entry)
            elif team == 'red_team' and phase == 'ban':
                red_bans.append(entry)
            elif team == 'red_team' and phase == 'pick':
                red_picks.append(entry)
        
        # Print results
        print(f"\nBLUE TEAM:")
        print(f"  Bans:  {', '.join(blue_bans) if blue_bans else 'None'}")
        print(f"  Picks: {', '.join(blue_picks) if blue_picks else 'None'}")
        
        print(f"\nRED TEAM:")
        print(f"  Bans:  {', '.join(red_bans) if red_bans else 'None'}")
        print(f"  Picks: {', '.join(red_picks) if red_picks else 'None'}")
        
        print(f"\nTotal: {len(detections)} champions detected")
        print("="*60 + "\n")
    
    def save_results(self, detections: List[Dict], output_path: str):
        """
        Save detection results to JSON file.
        
        Args:
            detections: List of detections
            output_path: Output file path
        """
        output_data = {
            'total_champions': len(detections),
            'champion_names': self.get_champion_names(detections),
            'detections': detections
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    def visualize_results(self, image_path: str, detections: List[Dict], 
                         output_path: str = None):
        """
        Draw bounding boxes and champion names on image.
        
        Args:
            image_path: Original image path
            detections: List of detections
            output_path: Output image path (if None, display only)
        """
        image = cv2.imread(str(image_path))
        
        # Colors for teams
        colors = {
            'blue_team': (255, 100, 0),   # Blue
            'red_team': (0, 50, 255),     # Red
            'unknown': (128, 128, 128)    # Gray
        }
        
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            name = det['champion_name']
            conf = det.get('confidence', 0)
            team = det.get('position', {}).get('team', 'unknown')
            
            # Get team color
            color = colors.get(team, colors['unknown'])
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(image, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Visualization saved to: {output_path}")
        else:
            cv2.imshow('Ban-Pick Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """
    Command-line interface for ban-pick detection.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect champions from League of Legends ban-pick screenshots'
    )
    parser.add_argument('image', type=str, help='Path to ban-pick image')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to YOLO model weights (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Save visualization to this path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display results in console')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BanPickDetector(yolo_model_path=args.model)
    
    # Detect champions
    detections = detector.detect_from_file(args.image)
    
    # Display results
    if not args.no_display:
        detector.print_results(detections)
    
    # Save JSON
    if args.output:
        detector.save_results(detections, args.output)
    
    # Visualize
    if args.visualize:
        detector.visualize_results(args.image, detections, args.visualize)
    
    # Print champion names
    champion_names = detector.get_champion_names(detections, top_n=10)
    print(f"\nTop 10 Champions:")
    for i, name in enumerate(champion_names, 1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
