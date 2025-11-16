#!/usr/bin/env python3
"""
Example script for multi-champion detection
Demonstrates YOLO + CLIP pipeline for detecting multiple champions
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from multi_champion_detector import MultiChampionDetector
from utils.image_utils import setup_logging

def create_test_image():
    """
    Create a simple test image with multiple champion portraits
    (For demonstration - in real use case, you'd have ban/pick screenshots)
    """
    
    # Create blank canvas (larger for better detection)
    canvas = np.ones((400, 800, 3), dtype=np.uint8) * 30
    
    # Load some test champion images and place them
    test_images = [
        "test_images/Ahri.jpg",
        "test_images/Yasuo.jpg", 
        "test_images/Jinx.jpg"
    ]
    
    # Larger positions and sizes for better YOLO detection
    positions = [(50, 50), (250, 50), (450, 50)]
    champion_size = (150, 150)  # Bigger size
    
    for i, (img_path, pos) in enumerate(zip(test_images, positions)):
        if Path(img_path).exists():
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to larger portrait size
                img_resized = cv2.resize(img, champion_size)
                x, y = pos
                h, w = champion_size[1], champion_size[0]
                
                # Add white border around each champion (helps with detection)
                cv2.rectangle(canvas, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), 3)
                
                # Place champion image
                canvas[y:y+h, x:x+w] = img_resized
                
                # Add champion name label
                cv2.putText(canvas, f"Champion {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save test image
    test_path = "multi_champion_test.jpg"
    cv2.imwrite(test_path, canvas)
    print(f"✅ Created test image: {test_path} with larger champion portraits")
    return test_path

def main():
    """Test multi-champion detection"""
    
    setup_logging()
    
    print("🎮 Multi-Champion Detection - Example")
    print("=" * 45)
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✅ Ultralytics installed")
    
    # Create or use existing test image
    test_image = "multi_champion_test.jpg"
    
    if not Path(test_image).exists():
        print("🎨 Creating test image with multiple champions...")
        test_image = create_test_image()
    
    print(f"🔍 Testing multi-detection on: {test_image}")
    
    try:
        # Initialize multi-champion detector
        print("🔧 Initializing Multi-Champion Detector (YOLO + CLIP)...")
        detector = MultiChampionDetector()
        
        # Run detection
        print("🎯 Running multi-champion detection...")
        results = detector.detect_champions(test_image, confidence_threshold=0.3)
        
        # Display results
        print(f"\n📊 Detection Results:")
        print("=" * 30)
        print(f"Total Champions Detected: {results['total_detected']}")
        print(f"YOLO Detections: {results['yolo_detections']}")
        
        if results['champions']:
            print(f"\n🏆 Detected Champions:")
            for i, champion in enumerate(results['champions'], 1):
                print(f"  {i}. {champion['champion_name']}")
                print(f"     CLIP Confidence: {champion['clip_confidence']:.3f}")
                print(f"     YOLO Confidence: {champion['yolo_confidence']:.3f}")
                print(f"     Bounding Box: {champion['bounding_box']}")
                print()
        else:
            print("❌ No champions detected")
            print("💡 This might happen because:")
            print("   - YOLO model is not trained for LoL champions yet")
            print("   - Need to adjust confidence threshold")
            print("   - Need custom YOLO training data")
        
        # Create visualization
        print("🎨 Creating annotated visualization...")
        annotated_path = detector.visualize_results(
            test_image, 
            results, 
            "multi_detection_example_annotated.jpg"
        )
        
        print(f"\n✨ Example completed!")
        print(f"📁 Files created:")
        print(f"   - {test_image} (test image)")
        print(f"   - {annotated_path} (annotated results)")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Train custom YOLO model on LoL champion icons")
        print(f"   2. Test on real ban/pick screenshots")
        print(f"   3. Use: python cli.py multi-detect your_banpick.jpg --visualize")
        
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   - Make sure ultralytics is installed: pip install ultralytics")
        print(f"   - Check GPU/CUDA availability")
        print(f"   - Verify test images exist in test_images/")
        raise

if __name__ == "__main__":
    main()