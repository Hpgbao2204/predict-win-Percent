#!/usr/bin/env python3
"""
Example script demonstrating Champion Detection System usage
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import with proper module path
import champion_detector
from champion_detector import ChampionDetectionSystem
import utils.image_utils

def main():
    """Run example detection"""
    
    # Setup logging
    utils.image_utils.setup_logging()
    
    print("🎮 League of Legends Champion Detection System - Example")
    print("=" * 60)
    
    # Initialize detection system
    print("🔧 Initializing detection system...")
    try:
        # Try to initialize with both CLIP and OpenAI
        detector = ChampionDetectionSystem(use_clip=True, use_openai=False)
        print("✅ Detection system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detection system: {str(e)}")
        return
    
    # Show system info
    info = detector.get_system_info()
    print(f"\n📊 System Info:")
    print(f"   Available methods: {', '.join(info['available_methods'])}")
    if info['clip_info']:
        print(f"   CLIP model: {info['clip_info']['model_name']}")
        print(f"   Champions loaded: {info['clip_info']['champions_loaded']}")
    
    # Example 1: Test with a skin from our database
    print(f"\n🧪 Example 1: Testing with known skin")
    test_skin_path = Path("../league-of-legends-skin-splash-art-collection/skins/Ahri/Ahri.jpg")
    
    if test_skin_path.exists():
        print(f"   Testing: {test_skin_path.name}")
        result = detector.detect_champion(test_skin_path, method="clip", save_results=False)
        
        if result["final_prediction"]:
            pred = result["final_prediction"]
            print(f"   🎯 Detected: {pred['champion_name']}")
            print(f"   📊 Confidence: {pred['confidence']:.3f}")
            print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
        else:
            print("   ❌ No detection")
    else:
        print(f"   ⚠️  Test skin not found: {test_skin_path}")
    
    # Example 2: Check test_images directory
    print(f"\n🧪 Example 2: Testing images in test_images/")
    test_dir = Path("test_images")
    
    if test_dir.exists():
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(test_dir.glob(f"*{ext}"))
        
        if image_files:
            print(f"   Found {len(image_files)} test images")
            
            for image_file in image_files[:3]:  # Test first 3 images
                print(f"   Testing: {image_file.name}")
                result = detector.detect_champion(image_file, method="clip", save_results=False)
                
                if result["final_prediction"]:
                    pred = result["final_prediction"]
                    print(f"     🎯 {pred['champion_name']} ({pred['confidence']:.3f})")
                else:
                    print(f"     ❌ No detection")
        else:
            print("   📁 No images found in test_images/")
            print("   💡 Add some champion images to test_images/ folder")
    else:
        print("   📁 test_images/ directory not found")
        print("   💡 Create test_images/ folder and add some champion images")
    
    # Example 3: Show top champions by skin count
    print(f"\n📈 Example 3: Champions with most skins")
    if detector.clip_detector and detector.clip_detector.champion_embeddings:
        champions_by_skins = []
        for champ_key, champ_data in detector.clip_detector.champion_embeddings.items():
            skin_count = len(champ_data["embeddings"])
            champions_by_skins.append((champ_data["name"], skin_count))
        
        champions_by_skins.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top 10 champions by skin count:")
        for i, (name, count) in enumerate(champions_by_skins[:10], 1):
            print(f"     {i:2d}. {name}: {count} skins")
    
    print(f"\n✨ Example completed!")
    print(f"\n💡 Try these commands:")
    print(f"   python cli.py info")
    print(f"   python cli.py detect your_image.jpg --method clip")
    print(f"   python cli.py batch your_image_folder/ --method auto")

if __name__ == "__main__":
    main()