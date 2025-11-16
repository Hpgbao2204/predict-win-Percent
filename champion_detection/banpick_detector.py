"""
League of Legends Ban/Pick Screen Champion Detection
Optimized for detecting 10 champions in ban/pick interface
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

from models.clip_detector import CLIPChampionDetector
from utils.image_utils import setup_logging

logger = logging.getLogger(__name__)

class BanPickDetector:
    """
    Specialized detector for LoL ban/pick screens
    Designed to find exactly 10 champions (5 bans + 5 picks per team)
    """
    
    def __init__(self):
        """Initialize ban/pick detector"""
        setup_logging()
        logger.info("Initializing Ban/Pick Champion Detector...")
        
        # Initialize CLIP for champion classification
        self.clip_detector = CLIPChampionDetector()
        
        logger.info("Ban/Pick Detector ready!")
    
    def detect_champions_in_banpick(self, image_path: str, 
                                   expected_champions: int = 10,
                                   confidence_threshold: float = 0.7) -> Dict:
        """
        Detect champions in ban/pick screen
        
        Args:
            image_path: Path to ban/pick screenshot
            expected_champions: Expected number of champions (usually 10)
            confidence_threshold: Minimum confidence for valid detection
            
        Returns:
            Dict with detection results
        """
        logger.info(f"Analyzing ban/pick screen: {image_path}")
        
        # Load and analyze image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Strategy 1: Grid-based detection for organized ban/pick layouts
        grid_results = self._detect_with_grid_strategy(image, confidence_threshold)
        
        # Strategy 2: Adaptive region detection
        adaptive_results = self._detect_with_adaptive_strategy(image, confidence_threshold)
        
        # Combine and optimize results
        final_results = self._combine_results([grid_results, adaptive_results], expected_champions)
        
        return {
            "champions": final_results,
            "total_detected": len(final_results),
            "image_path": image_path,
            "image_size": (width, height),
            "strategies_used": ["grid", "adaptive"]
        }
    
    def _detect_with_grid_strategy(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """
        Grid-based detection for standard ban/pick layouts
        Most LoL ban/pick screens have organized grid layouts
        """
        results = []
        height, width = image.shape[:2]
        
        # Common ban/pick dimensions (adjust based on your screenshots)
        # Usually 2 rows: bans on top, picks on bottom
        # Usually 5 columns for each team
        
        grid_configs = [
            # Config 1: 2x5 grid (bans + picks)
            {"rows": 2, "cols": 5, "margin": 0.1},
            # Config 2: 1x10 horizontal line
            {"rows": 1, "cols": 10, "margin": 0.15},
            # Config 3: 4x3 grid (with some empty spaces)
            {"rows": 3, "cols": 4, "margin": 0.1}
        ]
        
        for config in grid_configs:
            grid_results = self._scan_grid(image, config, confidence_threshold)
            results.extend(grid_results)
        
        return results
    
    def _scan_grid(self, image: np.ndarray, config: Dict, confidence_threshold: float) -> List[Dict]:
        """Scan image using grid configuration"""
        results = []
        height, width = image.shape[:2]
        
        rows, cols = config["rows"], config["cols"]
        margin = config["margin"]
        
        # Calculate cell dimensions with margins
        cell_width = int(width * (1 - 2 * margin) / cols)
        cell_height = int(height * (1 - 2 * margin) / rows)
        
        start_x = int(width * margin)
        start_y = int(height * margin)
        
        logger.info(f"Scanning {rows}x{cols} grid, cell size: {cell_width}x{cell_height}")
        
        for row in range(rows):
            for col in range(cols):
                # Calculate cell position
                x1 = start_x + col * cell_width
                y1 = start_y + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Extract cell region
                cell = image[y1:y2, x1:x2]
                
                if cell.size > 0:
                    # Classify champion in this cell
                    result = self._classify_region(cell, (x1, y1, x2, y2), f"grid_{row}_{col}")
                    
                    if result and result["confidence"] >= confidence_threshold:
                        result["detection_method"] = f"grid_{rows}x{cols}"
                        results.append(result)
        
        return results
    
    def _detect_with_adaptive_strategy(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """
        Adaptive detection using sliding window with optimized sizes
        Fallback for non-standard layouts
        """
        results = []
        height, width = image.shape[:2]
        
        # Optimized window sizes for champion portraits
        window_sizes = [
            (80, 80),   # Small portraits
            (120, 120), # Medium portraits
            (150, 150)  # Large portraits
        ]
        
        step_size = 40  # Smaller step for better coverage
        
        for win_w, win_h in window_sizes:
            for y in range(0, height - win_h, step_size):
                for x in range(0, width - win_w, step_size):
                    # Extract window
                    window = image[y:y+win_h, x:x+win_w]
                    
                    # Classify region
                    result = self._classify_region(window, (x, y, x+win_w, y+win_h), f"adaptive_{win_w}x{win_h}")
                    
                    if result and result["confidence"] >= confidence_threshold:
                        result["detection_method"] = f"adaptive_{win_w}x{win_h}"
                        results.append(result)
        
        return results
    
    def _classify_region(self, region: np.ndarray, bbox: Tuple[int, int, int, int], region_id: str) -> Optional[Dict]:
        """Classify a region using CLIP"""
        try:
            # Convert to RGB and create PIL image
            rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_region)
            
            # Save temporary region for CLIP
            temp_path = f"/tmp/region_{region_id}.jpg"
            pil_image.save(temp_path)
            
            # Get CLIP prediction
            clip_results = self.clip_detector.detect_champion(temp_path, top_k=1)
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            if clip_results and len(clip_results) > 0:
                prediction = clip_results[0]
                return {
                    "champion_name": prediction["champion_name"],
                    "confidence": prediction["confidence"],
                    "bounding_box": list(bbox),
                    "region_id": region_id
                }
        
        except Exception as e:
            logger.debug(f"Error classifying region {region_id}: {e}")
        
        return None
    
    def _combine_results(self, all_results: List[List[Dict]], expected_count: int) -> List[Dict]:
        """
        Combine results from multiple strategies and select best ones
        """
        # Flatten all results
        combined = []
        for results in all_results:
            combined.extend(results)
        
        if not combined:
            return []
        
        # Remove duplicates using Non-Maximum Suppression
        filtered = self._non_maximum_suppression(combined, iou_threshold=0.3)
        
        # Sort by confidence and take top N
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top results up to expected count
        return filtered[:expected_count]
    
    def _non_maximum_suppression(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self._calculate_iou(current["bounding_box"], det["bounding_box"]) < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_results(self, image_path: str, results: Dict, output_path: str = None) -> str:
        """Create annotated visualization of detection results"""
        image = cv2.imread(str(image_path))
        
        # Draw detections
        for i, champion in enumerate(results["champions"]):
            x1, y1, x2, y2 = champion["bounding_box"]
            
            # Color based on confidence
            confidence = champion["confidence"]
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw champion name and confidence
            label = f"{i+1}. {champion['champion_name']} ({confidence:.3f})"
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1-20), (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add summary
        summary = f"Detected: {results['total_detected']} champions"
        cv2.putText(image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save result
        if output_path is None:
            output_path = f"banpick_detection_{Path(image_path).stem}.jpg"
        
        cv2.imwrite(output_path, image)
        logger.info(f"Visualization saved: {output_path}")
        
        return output_path