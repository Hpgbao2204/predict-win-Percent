#!/usr/bin/env python3
"""
Debug sliding window detection to see what's happening
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))

from sliding_window_detector import SlidingWindowChampionDetector
from models.clip_detector import CLIPChampionDetector

def debug_single_window():
    """Debug detection on a single window"""
    
    print("🔍 Debug: Testing single champion detection")
    
    # Load the test image
    image = cv2.imread("multi_champion_test.jpg")
    if image is None:
        print("❌ Test image not found")
        return
    
    # Extract a specific champion region manually (we know Ahri is at ~50,50)
    x, y, w, h = 50, 50, 150, 150
    champion_crop = image[y:y+h, x:x+w]
    
    # Save crop for testing
    cv2.imwrite("debug_champion_crop.jpg", champion_crop)
    print(f"✅ Saved champion crop: debug_champion_crop.jpg")
    
    # Test with CLIP
    clip_detector = CLIPChampionDetector()
    result = clip_detector.detect_champion("debug_champion_crop.jpg")
    
    print(f"\n📊 CLIP Detection Results:")
    if result and len(result) > 0:
        for i, pred in enumerate(result[:3]):
            print(f"  {i+1}. {pred['champion_name']} - {pred['confidence']:.3f}")
    else:
        print("  ❌ No results from CLIP")

def debug_sliding_window():
    """Debug with very low threshold"""
    
    print(f"\n🔍 Debug: Sliding window with very low threshold")
    
    detector = SlidingWindowChampionDetector()
    
    results = detector.detect_champions(
        "multi_champion_test.jpg",
        window_sizes=[(150, 150)],  # Only one size for simplicity
        step_size=75,  # Larger step for faster testing
        confidence_threshold=0.3  # Very low threshold
    )
    
    print(f"Results: {results['total_detected']} detections")
    print(f"Windows scanned: {results['total_windows_scanned']}")

def main():
    debug_single_window()
    debug_sliding_window()

if __name__ == "__main__":
    main()