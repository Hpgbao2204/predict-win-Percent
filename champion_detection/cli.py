#!/usr/bin/env python3
"""
Command Line Interface for Champion Detection System
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from champion_detector import ChampionDetectionSystem
from utils.image_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(
        description="League of Legends Champion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect champion using CLIP
  python cli.py detect image.jpg --method clip
  
  # Detect using OpenAI Vision API  
  python cli.py detect image.jpg --method openai --api-key your_key
  
  # Batch detection on folder
  python cli.py batch /path/to/images --method auto
  
  # Get system information
  python cli.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect champion in single image')
    detect_parser.add_argument('image_path', help='Path to input image')
    detect_parser.add_argument('--method', 
                             choices=['clip', 'openai', 'auto', 'ensemble'],
                             default='auto',
                             help='Detection method to use (default: auto)')
    detect_parser.add_argument('--no-save', action='store_true',
                             help='Do not save results to file')
    detect_parser.add_argument('--api-key', help='OpenAI API key')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch detect champions in directory')
    batch_parser.add_argument('directory', help='Directory containing images')
    batch_parser.add_argument('--method',
                            choices=['clip', 'openai', 'auto', 'ensemble'], 
                            default='auto',
                            help='Detection method to use (default: auto)')
    batch_parser.add_argument('--pattern', default='*',
                            help='File pattern to match (default: *)')
    batch_parser.add_argument('--no-save', action='store_true',
                            help='Do not save results to file')
    batch_parser.add_argument('--api-key', help='OpenAI API key')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Multi-detect command (YOLO + CLIP)
    multi_parser = subparsers.add_parser('multi-detect', help='Detect multiple champions in single image (YOLO + CLIP)')
    multi_parser.add_argument('image_path', help='Path to input image (e.g., ban/pick screen)')
    multi_parser.add_argument('--confidence', type=float, default=0.5,
                            help='YOLO confidence threshold (default: 0.5)')
    multi_parser.add_argument('--visualize', action='store_true',
                            help='Create annotated output image')
    multi_parser.add_argument('--output', help='Output path for annotated image')
    
    # Ban/Pick detection command  
    banpick_parser = subparsers.add_parser('banpick', help='Detect champions in ban/pick screen (optimized for 10 champions)')
    banpick_parser.add_argument('image_path', help='Path to ban/pick screenshot')
    banpick_parser.add_argument('--confidence', type=float, default=0.7,
                               help='Confidence threshold (default: 0.7)')
    banpick_parser.add_argument('--expected', type=int, default=10,
                               help='Expected number of champions (default: 10)')
    banpick_parser.add_argument('--visualize', action='store_true',
                               help='Create annotated output image')
    banpick_parser.add_argument('--output', help='Output path for results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--api-key', help='OpenAI API key')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging()
    
    # Set API key if provided
    if hasattr(args, 'api_key') and args.api_key:
        import os
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    try:
        if args.command == 'detect':
            handle_detect_command(args)
        elif args.command == 'multi-detect':
            handle_multi_detect_command(args)
        elif args.command == 'banpick':
            handle_banpick_command(args)
        elif args.command == 'batch':
            handle_batch_command(args)
        elif args.command == 'info':
            handle_info_command(args)
        elif args.command == 'test':
            handle_test_command(args)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_detect_command(args):
    """Handle single image detection"""
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Detecting champion in: {image_path}")
    print(f"Method: {args.method}")
    print("-" * 50)
    
    # Determine which detectors to use
    use_clip = args.method in ['clip', 'auto', 'ensemble']
    use_openai = args.method in ['openai', 'auto', 'ensemble']
    
    # Initialize detection system
    detector = ChampionDetectionSystem(use_clip=use_clip, use_openai=use_openai)
    
    # Run detection
    results = detector.detect_champion(
        image_path, 
        method=args.method, 
        save_results=not args.no_save
    )
    
    # Display results
    print_detection_results(results)

def handle_batch_command(args):
    """Handle batch detection"""
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    print(f"Batch detection in: {directory}")
    print(f"Method: {args.method}")
    print(f"Pattern: {args.pattern}")
    print("-" * 50)
    
    # Determine which detectors to use
    use_clip = args.method in ['clip', 'auto', 'ensemble']
    use_openai = args.method in ['openai', 'auto', 'ensemble']
    
    # Initialize detection system
    detector = ChampionDetectionSystem(use_clip=use_clip, use_openai=use_openai)
    
    # Run batch detection
    results = detector.batch_detect(
        directory,
        method=args.method,
        pattern=args.pattern,
        save_batch_results=not args.no_save
    )
    
    # Display summary
    print_batch_summary(results)

def handle_info_command(args):
    """Handle system info command"""
    print("Champion Detection System Information")
    print("=" * 50)
    
    try:
        # Initialize system with both methods if possible
        detector = ChampionDetectionSystem(use_clip=True, use_openai=True)
    except:
        try:
            # Fallback to CLIP only
            detector = ChampionDetectionSystem(use_clip=True, use_openai=False)
        except:
            print("Error: Could not initialize detection system")
            return
    
    info = detector.get_system_info()
    print_system_info(info)

def handle_test_command(args):
    """Handle test command"""
    print("Running Champion Detection System Tests")
    print("=" * 50)
    
    # Test with sample images if available
    test_dir = Path(__file__).parent / "test_images"
    
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print("No test images found. Please add some test images to:")
        print(f"  {test_dir}")
        print("\nYou can copy some champion splash arts there for testing.")
        return
    
    # Run tests
    detector = ChampionDetectionSystem(use_clip=True, use_openai=True)
    results = detector.batch_detect(test_dir, save_batch_results=False)
    
    print(f"Tested {len(results)} images")
    successful = sum(1 for r in results if r["final_prediction"])
    print(f"Successful detections: {successful}/{len(results)}")

def print_detection_results(results):
    """Print formatted detection results"""
    print("Detection Results:")
    print("=" * 30)
    
    if results["final_prediction"]:
        pred = results["final_prediction"]
        print(f"🎯 Champion: {pred['champion_name']}")
        print(f"📊 Confidence: {pred['confidence']:.3f}")
        
        if "reasoning" in pred:
            print(f"💭 Reasoning: {pred['reasoning']}")
        
        if "best_matching_skin" in pred:
            print(f"🎨 Best Match: {pred['best_matching_skin']}")
    else:
        print("❌ No champion detected")
    
    print(f"\n⏱️ Processing Time: {results['processing_time']:.2f} seconds")
    
    # Show detailed results if available
    if "clip" in results["results"] and results["results"]["clip"]:
        print(f"\n🔍 CLIP Top Results:")
        for i, result in enumerate(results["results"]["clip"][:3], 1):
            print(f"  {i}. {result['champion_name']} ({result['confidence']:.3f})")
    
    if "openai" in results["results"] and results["results"]["openai"]:
        openai_result = results["results"]["openai"]
        print(f"\n🤖 OpenAI Vision Result:")
        print(f"  Champion: {openai_result['champion_name']}")
        print(f"  Confidence: {openai_result['confidence']:.3f}")

def print_batch_summary(results):
    """Print batch detection summary"""
    total = len(results)
    successful = sum(1 for r in results if r["final_prediction"])
    
    print(f"\nBatch Detection Summary:")
    print("=" * 30)
    print(f"📸 Total Images: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {total - successful}")
    print(f"📈 Success Rate: {(successful/total)*100:.1f}%")
    
    # Show some examples
    print(f"\n🎯 Sample Detections:")
    for i, result in enumerate(results[:5]):
        if result["final_prediction"]:
            pred = result["final_prediction"]
            filename = Path(result["input_image"]).name
            print(f"  {filename} → {pred['champion_name']} ({pred['confidence']:.3f})")

def print_system_info(info):
    """Print system information"""
    print(f"Available Methods: {', '.join(info['available_methods'])}")
    print(f"OpenAI Available: {'Yes' if info['openai_available'] else 'No'}")
    
    if info['clip_info']:
        clip = info['clip_info']
        print(f"\nCLIP Model Info:")
        print(f"  Model: {clip['model_name']}")
        print(f"  Device: {clip['device']}")
        print(f"  Champions: {clip['champions_loaded']}")
        print(f"  Total Embeddings: {clip['total_embeddings']}")
    
    settings = info['settings']
    print(f"\nSettings:")
    print(f"  Similarity Threshold: {settings['similarity_threshold']}")
    print(f"  Top-K Results: {settings['top_k_results']}")
    print(f"  Max Image Size: {settings['max_image_size']}")

def handle_multi_detect_command(args):
    """Handle multi-champion detection command"""
    from multi_champion_detector import MultiChampionDetector
    
    print(f"Multi-Champion Detection")
    print(f"Image: {args.image_path}")
    print(f"YOLO Confidence: {args.confidence}")
    print("-" * 50)
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return
    
    try:
        # Initialize multi-champion detector
        detector = MultiChampionDetector()
        
        # Run detection
        results = detector.detect_champions(args.image_path, args.confidence)
        
        # Display results
        print(f"\n🎯 Detection Results:")
        print("=" * 40)
        print(f"📊 Total Champions Detected: {results['total_detected']}")
        print(f"🔍 YOLO Detections: {results['yolo_detections']}")
        
        if results['champions']:
            print(f"\n📋 Detected Champions:")
            for i, champ in enumerate(results['champions'], 1):
                print(f"  {i}. {champ['champion_name']}")
                print(f"     CLIP Confidence: {champ['clip_confidence']:.3f}")
                print(f"     YOLO Confidence: {champ['yolo_confidence']:.3f}")
                print(f"     Bounding Box: {champ['bounding_box']}")
                print()
        else:
            print("❌ No champions detected")
        
        # Create visualization if requested
        if args.visualize:
            output_path = args.output or f"multi_detection_{Path(args.image_path).stem}_annotated.jpg"
            annotated_path = detector.visualize_results(args.image_path, results, output_path)
            print(f"🎨 Annotated image saved: {annotated_path}")
        
        # Save results
        results_file = f"multi_detection_results_{Path(args.image_path).stem}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved: {results_file}")
        
    except Exception as e:
        print(f"❌ Detection failed: {str(e)}")
        raise

def handle_banpick_command(args):
    """Handle ban/pick detection command"""
    from template_champion_detector import TemplateChampionDetector
    
    print(f"🎮 Ban/Pick Champion Detection")
    print(f"Image: {args.image_path}")
    print(f"Expected Champions: {args.expected}")
    print(f"Confidence: {args.confidence}")
    print("-" * 50)
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return
    
    try:
        # Initialize Template Matching detector  
        detector = TemplateChampionDetector()
        
        # Run detection
        results = detector.detect_champions(
            args.image_path,
            layout="auto",
            confidence_threshold=args.confidence
        )
        
        # Display results
        print(f"\n🎯 Detection Results:")
        print("=" * 40)
        print(f"📊 Champions Detected: {results['total_detected']}/10")
        print(f"📐 Image Size: {results['image_size']}")
        print(f"🔍 Template Regions: {results['card_regions_found']}")
        print(f"📋 Layout Used: {results['layout_used']}")
        
        if results['champions']:
            print(f"\n📋 Detected Champions:")
            for i, champ in enumerate(results['champions'], 1):
                print(f"  {i:2d}. {champ['champion_name']:<15} (Confidence: {champ['confidence']:.3f}, Method: {champ['detection_method']})")
            
            # Show unique champions
            unique_champions = list(set(c['champion_name'] for c in results['champions']))
            print(f"\n🏆 Unique Champions Found: {len(unique_champions)}")
            print(f"    {', '.join(unique_champions)}")
            
        else:
            print("❌ No champions detected")
            print("💡 Try lowering --confidence or check image quality")
        
        # Create visualization
        if args.visualize:
            output_path = args.output or f"banpick_{Path(args.image_path).stem}_result.jpg"
            annotated_path = detector.visualize_results(args.image_path, results, output_path)
            print(f"\n🎨 Annotated result: {annotated_path}")
        
        # Save results
        results_file = f"banpick_results_{Path(args.image_path).stem}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved: {results_file}")
        
    except Exception as e:
        print(f"❌ Detection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()