#!/usr/bin/env python3
"""
Simple example to test ban/pick champion detection
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from banpick_detector import BanPickDetector

def main():
    """Test ban/pick detection with the test image"""
    
    print("🎮 Simple Ban/Pick Detection Test")
    print("=" * 40)
    
    # Test image path
    test_image = "../test_images/test_10champ.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("💡 Make sure test_10champ.png is in test_images/ folder")
        return
    
    try:
        # Initialize detector
        print("🔧 Loading detection system...")
        detector = BanPickDetector()
        
        # Run detection with optimized settings
        print("🎯 Analyzing ban/pick screen...")
        results = detector.detect_champions_in_banpick(
            test_image,
            expected_champions=10,
            confidence_threshold=0.65  # Slightly higher threshold
        )
        
        print(f"\n📊 Results Summary:")
        print(f"   Total detections: {results['total_detected']}")
        print(f"   Image size: {results['image_size']}")
        
        if results['champions']:
            # Get unique champions (remove duplicates)
            seen_champions = {}
            unique_champions = []
            
            for champ in results['champions']:
                name = champ['champion_name']
                if name not in seen_champions or champ['confidence'] > seen_champions[name]['confidence']:
                    seen_champions[name] = champ
            
            unique_champions = list(seen_champions.values())
            unique_champions.sort(key=lambda x: x['confidence'], reverse=True)
            
            print(f"\n🏆 Unique Champions (Top {min(10, len(unique_champions))}):")
            for i, champ in enumerate(unique_champions[:10], 1):
                print(f"   {i:2d}. {champ['champion_name']:<15} ({champ['confidence']:.3f})")
        
        # Create visualization
        print(f"\n🎨 Creating visualization...")
        output_path = detector.visualize_results(test_image, results, "simple_banpick_result.jpg")
        
        print(f"\n✨ Test completed!")
        print(f"📁 Check result: {output_path}")
        print(f"\n💡 Usage:")
        print(f"   python cli.py banpick your_banpick_screenshot.png --visualize")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()