#!/usr/bin/env python3
"""
Test sliding window approach for multi-champion detection
This works without needing custom YOLO training
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sliding_window_detector import SlidingWindowChampionDetector
from utils.image_utils import setup_logging

def main():
    """Test sliding window multi-champion detection"""
    
    setup_logging()
    
    print("🎮 Sliding Window Multi-Champion Detection - Test")
    print("=" * 55)
    
    # Use existing test image
    test_image = "multi_champion_test.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("💡 Run example_multi_detection.py first to create test image")
        return
    
    print(f"🔍 Testing sliding window detection on: {test_image}")
    
    try:
        # Initialize sliding window detector
        print("🔧 Initializing Sliding Window Detector...")
        detector = SlidingWindowChampionDetector()
        
        # Run detection with optimized parameters
        print("🎯 Running sliding window detection...")
        print("   This may take a moment as we scan the entire image...")
        
        results = detector.detect_champions(
            test_image,
            window_sizes=[(120, 120), (150, 150), (180, 180)],  # Different sizes
            step_size=30,  # Smaller step for better coverage
            confidence_threshold=0.7  # Lower threshold for testing
        )
        
        # Display results
        print(f"\n📊 Detection Results:")
        print("=" * 35)
        print(f"Total Champions Detected: {results['total_detected']}")
        print(f"Windows Scanned: {results['total_windows_scanned']}")
        
        if results['champions']:
            print(f"\n🏆 Detected Champions:")
            for i, champion in enumerate(results['champions'], 1):
                print(f"  {i}. {champion['champion_name']}")
                print(f"     Confidence: {champion['confidence']:.3f}")
                print(f"     Position: {champion['bounding_box']}")
                print(f"     Window Size: {champion['window_size']}")
                print()
        else:
            print("❌ No champions detected")
            print("💡 Try lowering confidence_threshold or adjusting window sizes")
        
        # Create visualization
        print("🎨 Creating annotated visualization...")
        annotated_path = detector.visualize_results(
            test_image, 
            results, 
            "sliding_window_detection_annotated.jpg"
        )
        
        print(f"\n✨ Sliding window test completed!")
        print(f"📁 Files created:")
        print(f"   - {annotated_path} (annotated results)")
        
        print(f"\n💡 Performance Notes:")
        print(f"   - Sliding window is slower but works without training")
        print(f"   - For production: train custom YOLO model on LoL champions")
        print(f"   - Current approach good for proof of concept")
        
        print(f"\n🚀 Usage:")
        print(f"   python cli.py sliding-detect your_image.jpg --confidence 0.7")
        
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        raise

if __name__ == "__main__":
    main()