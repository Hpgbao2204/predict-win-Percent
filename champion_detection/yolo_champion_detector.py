"""""""""

YOLOv8 + CLIP Multi-Champion Detection Pipeline

Complete solution: YOLO detects champion cards -> CLIP classifies each cropYOLOv8 + CLIP Multi-Champion Detection PipelineYOLOv8 + CLIP Multi-Champion Detection Pipeline

"""

Complete solution: YOLO detects champion cards → CLIP classifies each cropCORRECT approach: YOLO detects champion cards → CLIP classifies each crop

import cv2

import numpy as np""""""

from typing import List, Dict, Tuple, Optional

import logging

from pathlib import Path

import jsonimport cv2import cv2

import time

import numpy as npimport numpy as np

from models.yolo.card_detector import YOLOCardDetector

from models.clip.champion_classifier import CLIPChampionClassifierfrom typing import List, Dict, Tuple, Optionalfrom typing import List, Dict, Tuple, Optional

from utils.image_utils import setup_logging

import loggingimport logging

logger = logging.getLogger(__name__)

from pathlib import Pathfrom pathlib import Path

class YOLOChampionDetector:

    """import jsonimport json

    Complete YOLO + CLIP Pipeline for Champion Detection

    import time

    Pipeline:

    1. YOLO detects champion card bounding boxesfrom models.yolo.card_detector import YOLOCardDetector

    2. CLIP classifies each cropped champion

    3. Return complete results with champion namesfrom models.yolo.card_detector import YOLOCardDetectorfrom models.clip.champion_classifier import CLIPChampionClassifier

    """

    from models.clip.champion_classifier import CLIPChampionClassifierfrom utils.image_utils import setup_logging

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):

        """Initialize YOLO + CLIP pipeline"""from utils.image_utils import setup_logging

        setup_logging()

        logger.info("🚀 Initializing YOLO + CLIP Multi-Champion Detector")logger = logging.getLogger(__name__)

        

        # Initialize YOLO detectorlogger = logging.getLogger(__name__)

        logger.info("🎯 Loading YOLO card detector...")

        self.yolo_detector = YOLOCardDetector(yolo_model_path)class YOLOChampionDetector:

        

        # Initialize CLIP classifierclass YOLOChampionDetector:    """

        logger.info("📋 Loading CLIP classifier...")

        self.clip_classifier = CLIPChampionClassifier()    """    Complete YOLO + CLIP Pipeline for Champion Detection

        

        logger.info("✅ YOLO + CLIP pipeline ready!")    Complete YOLO + CLIP Pipeline for Champion Detection    

    

    def detect_champions(self,         Pipeline:

                        image: np.ndarray,

                        confidence_threshold: float = 0.5,    Pipeline:    1. YOLO detects champion card bounding boxes

                        top_k_predictions: int = 3) -> List[Dict]:

        """    1. YOLO detects champion card bounding boxes    2. CLIP classifies each cropped champion

        Complete champion detection pipeline

            2. CLIP classifies each cropped champion    3. Return complete results with champion names

        Args:

            image: Input image (BGR format)    3. Return complete results with champion names    """

            confidence_threshold: YOLO detection threshold

            top_k_predictions: Number of CLIP predictions per detection    """    

            

        Returns:        def __init__(self, yolo_model_path: str = "yolov8n.pt"):

            List of champion detections with full information

        """    def __init__(self, yolo_model_path: str = "yolov8n.pt"):        """Initialize YOLO + CLIP pipeline"""

        

        start_time = time.time()        """Initialize YOLO + CLIP pipeline"""        setup_logging()

        

        # Step 1: YOLO detection        setup_logging()        logger.info("🚀 Initializing YOLO + CLIP Multi-Champion Detector")

        logger.info("🎯 Running YOLO champion card detection...")

        detections = self.yolo_detector.detect_and_crop(image, confidence_threshold)        logger.info("🚀 Initializing YOLO + CLIP Multi-Champion Detector")        

        

        if not detections:                # Initialize YOLO detector

            logger.warning("No champion cards detected by YOLO")

            return []        # Initialize YOLO detector        logger.info("🎯 Loading YOLO card detector...")

        

        detection_time = time.time() - start_time        logger.info("🎯 Loading YOLO card detector...")        self.yolo_detector = YOLOCardDetector(yolo_model_path)

        logger.info(f"YOLO detected {len(detections)} cards in {detection_time:.3f}s")

                self.yolo_detector = YOLOCardDetector(yolo_model_path)        

        # Step 2: CLIP classification

        logger.info("🔍 Running CLIP champion classification...")                # Initialize CLIP classifier

        classified_detections = self.clip_classifier.classify_detections(

            detections, top_k=top_k_predictions        # Initialize CLIP classifier        logger.info("📋 Loading CLIP classifier...")

        )

                logger.info("📋 Loading CLIP classifier...")        self.clip_classifier = CLIPChampionClassifier()

        total_time = time.time() - start_time

        logger.info(f"✅ Complete pipeline finished in {total_time:.3f}s")        self.clip_classifier = CLIPChampionClassifier()        

        

        return classified_detections                logger.info("✅ YOLO + CLIP pipeline ready!")

    

    def detect_champions_simple(self,         logger.info("✅ YOLO + CLIP pipeline ready!")    

                              image: np.ndarray,

                              confidence_threshold: float = 0.5) -> List[str]:        def _load_yolo_model(self, model_path: Optional[str] = None) -> YOLO:

        """

        Simplified detection - just return champion names    def detect_champions(self,         """Load YOLO model for detecting champion cards/portraits"""

        

        Args:                        image: np.ndarray,        

            image: Input image

            confidence_threshold: Detection threshold                        confidence_threshold: float = 0.5,        if model_path and Path(model_path).exists():

            

        Returns:                        top_k_predictions: int = 3) -> List[Dict]:            # Load custom trained YOLO model (future)

            List of detected champion names (best predictions only)

        """        """            logger.info(f"Loading custom YOLO model: {model_path}")

        

        detections = self.detect_champions(image, confidence_threshold, top_k_predictions=1)        Complete champion detection pipeline            model = YOLO(model_path)

        

        champion_names = []                else:

        for detection in detections:

            champion = detection.get("champion", "Unknown")        Args:            # Use pretrained YOLOv8 - it can detect people/faces which works for champion portraits

            if champion != "Unknown":

                champion_names.append(champion)            image: Input image (BGR format)            logger.info("Loading YOLOv8n for general object detection...")

        

        return champion_names            confidence_threshold: YOLO detection threshold            model = YOLO('yolov8n.pt')



            top_k_predictions: Number of CLIP predictions per detection        

def main():

    """Test the complete YOLO + CLIP pipeline"""                    return model

    

    # Initialize detector        Returns:    

    detector = YOLOChampionDetector()

                List of champion detections with full information:    def detect_champions(self, image_path: str, 

    # Test image path

    test_image_path = "test_images/banpick_sample.jpg"            [                        confidence_threshold: float = 0.3,

    

    if Path(test_image_path).exists():                {                        expected_champions: int = 10) -> Dict:

        logger.info(f"📸 Loading test image: {test_image_path}")

                            "bbox": [x1, y1, x2, y2],        """

        # Load image

        image = cv2.imread(test_image_path)                    "confidence": float,  # YOLO confidence        Main detection pipeline: YOLO → Crop → CLIP

        

        # Run complete detection                    "champion": str,      # Best champion prediction        

        detections = detector.detect_champions(image)

                            "champion_confidence": float,  # CLIP confidence        Args:

        # Print results

        print(f"\\n🎯 Detection Results:")                    "predictions": [      # Top-k CLIP predictions            image_path: Path to ban/pick screenshot

        print(f"Total detections: {len(detections)}")

                                {"champion": str, "confidence": float}, ...            confidence_threshold: YOLO detection confidence

        for i, detection in enumerate(detections):

            print(f"\\nCard {i+1}:")                    ],            expected_champions: Expected number of champions to find

            print(f"  Champion: {detection['champion']}")

            print(f"  CLIP Confidence: {detection['champion_confidence']:.3f}")                    "crop": np.ndarray   # Cropped champion image            

            print(f"  YOLO Confidence: {detection['confidence']:.3f}")

            print(f"  BBox: {detection['bbox']}")                },        Returns:

        

    else:                ...            Detection results with champion names and positions

        print(f"❌ Test image not found: {test_image_path}")

        print("Pipeline initialized successfully! Ready for detection.")            ]        """



        """        logger.info(f"🔍 Processing ban/pick screen: {image_path}")

if __name__ == "__main__":

    main()                

        start_time = time.time()        # Load image

                image = cv2.imread(str(image_path))

        # Step 1: YOLO detection        if image is None:

        logger.info("🎯 Running YOLO champion card detection...")            raise ValueError(f"Could not load image: {image_path}")

        detections = self.yolo_detector.detect_and_crop(image, confidence_threshold)        

                height, width = image.shape[:2]

        if not detections:        logger.info(f"📐 Image dimensions: {width}x{height}")

            logger.warning("No champion cards detected by YOLO")        

            return []        # Step 1: YOLO detection to find champion card regions

                logger.info("🎯 Step 1: Running YOLO object detection...")

        detection_time = time.time() - start_time        yolo_detections = self._run_yolo_detection(image, confidence_threshold)

        logger.info(f"YOLO detected {len(detections)} cards in {detection_time:.3f}s")        

                if len(yolo_detections) < expected_champions:

        # Step 2: CLIP classification            logger.warning(f"⚠️ YOLO detected {len(yolo_detections)}, expected {expected_champions}")

        logger.info("🔍 Running CLIP champion classification...")            # Fallback: use intelligent grid to fill missing detections

        classified_detections = self.clip_classifier.classify_detections(            fallback_detections = self._fallback_intelligent_grid(image, expected_champions)

            detections, top_k=top_k_predictions            yolo_detections.extend(fallback_detections)

        )            # Remove duplicates and keep best

                    yolo_detections = yolo_detections[:expected_champions]

        total_time = time.time() - start_time        

        logger.info(f"✅ Complete pipeline finished in {total_time:.3f}s")        logger.info(f"📊 YOLO found {len(yolo_detections)} potential champion regions")

                

        return classified_detections        # Step 2: CLIP classification for each detected region

            logger.info("📋 Step 2: CLIP classification of detected regions...")

    def detect_champions_simple(self,         champion_results = []

                              image: np.ndarray,        

                              confidence_threshold: float = 0.5) -> List[str]:        for i, detection in enumerate(yolo_detections):

        """            # Crop the detected champion region

        Simplified detection - just return champion names            cropped_champion = self._crop_champion_region(image, detection)

                    

        Args:            # Classify the cropped region with CLIP

            image: Input image            clip_result = self._classify_champion_crop(cropped_champion, f"region_{i}")

            confidence_threshold: Detection threshold            

                        if clip_result:

        Returns:                combined_result = {

            List of detected champion names (best predictions only)                    "champion_name": clip_result["champion_name"],

        """                    "clip_confidence": clip_result["confidence"],

                            "yolo_confidence": detection["confidence"],

        detections = self.detect_champions(image, confidence_threshold, top_k_predictions=1)                    "bounding_box": detection["bbox"],

                            "region_id": i,

        champion_names = []                    "detection_method": "yolo+clip"

        for detection in detections:                }

            champion = detection.get("champion", "Unknown")                champion_results.append(combined_result)

            if champion != "Unknown":        

                champion_names.append(champion)        # Step 3: Post-process results

                final_results = self._post_process_results(champion_results, expected_champions)

        return champion_names        

            logger.info(f"🎉 Final result: {len(final_results)} champions detected")

    def analyze_banpick(self, image: np.ndarray) -> Dict:        

        """        return {

        Analyze a complete ban/pick screenshot            "champions": final_results,

                    "total_detected": len(final_results),

        Args:            "yolo_detections": len(yolo_detections),

            image: Ban/pick screenshot            "image_path": image_path,

                        "image_size": (width, height)

        Returns:        }

            Complete analysis with team compositions and statistics    

        """    def _run_yolo_detection(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:

                """

        logger.info("🏟️ Analyzing ban/pick composition...")        Run YOLO detection to find champion card/portrait regions

                

        # Detect all champions        Strategy: Use YOLO's person detection since champion portraits contain human-like figures

        detections = self.detect_champions(image, confidence_threshold=0.4)        """

                

        # Get summary        # Run YOLO inference

        summary = self.clip_classifier.get_champion_summary(detections)        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)

                

        # Organize results        detections = []

        analysis = {        for result in results:

            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),            boxes = result.boxes

            "total_detections": len(detections),            if boxes is not None:

            "champions_detected": summary["detected_champions"],                for i in range(len(boxes)):

            "unique_champions": len(summary["champions"]),                    # Get detection info

            "average_confidence": summary["average_confidence"],                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

            "champion_details": summary["champions"],                    confidence = float(boxes.conf[i].cpu().numpy())

            "detections": detections                    class_id = int(boxes.cls[i].cpu().numpy())

        }                    class_name = self.yolo_model.names[class_id]

                            

        logger.info(f"📊 Analysis complete: {analysis['total_detections']} cards, "                    # Filter for relevant object types

                   f"{analysis['unique_champions']} unique champions")                    # Accept: person (0), sports ball (32), books (73), etc.

                            # These often trigger on champion portraits and UI elements

        return analysis                    relevant_classes = [0, 32, 67, 73, 74, 76]  # person, sports ball, cell phone, book, clock, keyboard

                        

    def visualize_results(self,                     if class_id in relevant_classes:

                         image: np.ndarray,                        width = x2 - x1

                         detections: List[Dict],                        height = y2 - y1

                         show_predictions: bool = True) -> np.ndarray:                        

        """                        # Filter by reasonable champion card size

        Create visualization of detection results                        if (30 < width < image.shape[1] * 0.8 and 

                                    30 < height < image.shape[0] * 0.8 and

        Args:                            width * height > 1000):  # Minimum area

            image: Original image                            

            detections: Detection results from detect_champions()                            detections.append({

            show_predictions: Whether to show champion predictions                                "bbox": [int(x1), int(y1), int(x2), int(y2)],

                                            "confidence": confidence,

        Returns:                                "class_id": class_id,

            Image with bounding boxes and champion labels                                "class_name": class_name,

        """                                "area": width * height

                                    })

        vis_image = image.copy()        

                # Sort by confidence and area (larger, more confident detections first)

        for i, detection in enumerate(detections):        detections.sort(key=lambda x: (x["confidence"], x["area"]), reverse=True)

            x1, y1, x2, y2 = detection["bbox"]        

                    return detections

            # Draw bounding box    

            color = (0, 255, 0)  # Green    def _fallback_intelligent_grid(self, image: np.ndarray, expected_champions: int) -> List[Dict]:

            thickness = 2        """

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)        Intelligent fallback when YOLO doesn't detect enough regions

                    Creates smart grid based on typical ban/pick layouts

            if show_predictions:        """

                # Prepare label        height, width = image.shape[:2]

                champion = detection.get("champion", "Unknown")        logger.info("🔄 Using intelligent grid fallback...")

                clip_conf = detection.get("champion_confidence", 0.0)        

                yolo_conf = detection.get("confidence", 0.0)        # Common LoL ban/pick layouts

                        if expected_champions == 10:

                label = f"{champion}"            # 2 rows of 5 (bans + picks)

                conf_label = f"CLIP: {clip_conf:.2f} | YOLO: {yolo_conf:.2f}"            if width > height * 2:  # Wide image

                                rows, cols = 2, 5

                # Draw champion name            else:  # More square image  

                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]                rows, cols = 1, 10

                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 35),         else:

                            (x1 + max(label_size[0], 200), y1), color, -1)            # Dynamic grid based on expected count

                cv2.putText(vis_image, label, (x1 + 2, y1 - 20),             cols = min(expected_champions, 10)

                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)            rows = max(1, expected_champions // cols)

                        

                # Draw confidence scores        detections = []

                cv2.putText(vis_image, conf_label, (x1 + 2, y1 - 5),         

                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)        # Calculate grid with margins

                margin_x, margin_y = width * 0.05, height * 0.1

        return vis_image        cell_width = (width - 2 * margin_x) / cols

            cell_height = (height - 2 * margin_y) / rows

    def save_results(self,         

                    detections: List[Dict],        for row in range(rows):

                    output_path: str,            for col in range(cols):

                    include_crops: bool = False):                if len(detections) >= expected_champions:

        """                    break

        Save detection results to JSON file                

                        x1 = int(margin_x + col * cell_width)

        Args:                y1 = int(margin_y + row * cell_height)

            detections: Detection results                x2 = int(x1 + cell_width * 0.8)  # Leave some gap

            output_path: Output JSON file path                y2 = int(y1 + cell_height * 0.8)

            include_crops: Whether to save crop images separately                

        """                detections.append({

                            "bbox": [x1, y1, x2, y2],

        # Prepare data for JSON (remove numpy arrays)                    "confidence": 0.8,  # High confidence for fallback

        json_data = []                    "class_id": -1,

                            "class_name": "fallback_grid",

        for i, detection in enumerate(detections):                    "area": (x2-x1) * (y2-y1)

            json_detection = {                })

                "id": i,        

                "bbox": detection["bbox"],        return detections

                "confidence": float(detection["confidence"]),    

                "champion": detection.get("champion", "Unknown"),    def _crop_champion_region(self, image: np.ndarray, detection: Dict) -> np.ndarray:

                "champion_confidence": float(detection.get("champion_confidence", 0.0)),        """Crop champion region with smart padding"""

                "predictions": detection.get("predictions", [])        x1, y1, x2, y2 = detection["bbox"]

            }        

                    # Add small padding to ensure we get the full champion

            # Save crop image separately if requested        padding = 10

            if include_crops and "crop" in detection:        x1 = max(0, x1 - padding)

                crop_filename = f"crop_{i:03d}.jpg"        y1 = max(0, y1 - padding)

                crop_path = Path(output_path).parent / crop_filename        x2 = min(image.shape[1], x2 + padding)

                cv2.imwrite(str(crop_path), detection["crop"])        y2 = min(image.shape[0], y2 + padding)

                json_detection["crop_file"] = crop_filename        

                    cropped = image[y1:y2, x1:x2]

            json_data.append(json_detection)        return cropped

            

        # Save JSON    def _classify_champion_crop(self, cropped_image: np.ndarray, crop_id: str) -> Optional[Dict]:

        with open(output_path, 'w') as f:        """Classify champion using CLIP on clean cropped region"""

            json.dump({        

                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),        try:

                "total_detections": len(detections),            # Convert BGR to RGB for CLIP

                "detections": json_data            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            }, f, indent=2)            pil_image = Image.fromarray(rgb_image)

                    

        logger.info(f"💾 Results saved to: {output_path}")            # Save temp crop for CLIP

            temp_path = f"/tmp/champion_crop_{crop_id}.jpg"

            pil_image.save(temp_path, quality=95)

def main():            

    """Test the complete YOLO + CLIP pipeline"""            # Get CLIP prediction

                clip_results = self.clip_detector.detect_champion(temp_path, top_k=1)

    # Initialize detector            

    detector = YOLOChampionDetector()            # Clean up

                Path(temp_path).unlink(missing_ok=True)

    # Test image path            

    test_image_path = "test_images/banpick_sample.jpg"            if clip_results and len(clip_results) > 0:

                    prediction = clip_results[0]

    if Path(test_image_path).exists():                return {

        logger.info(f"📸 Loading test image: {test_image_path}")                    "champion_name": prediction["champion_name"],

                            "confidence": prediction["confidence"]

        # Load image                }

        image = cv2.imread(test_image_path)        

                except Exception as e:

        # Run complete detection            logger.error(f"❌ Error classifying crop {crop_id}: {e}")

        detections = detector.detect_champions(image)        

                return None

        # Print results    

        print(f"\n🎯 Detection Results:")    def _post_process_results(self, results: List[Dict], expected_count: int) -> List[Dict]:

        print(f"Total detections: {len(detections)}")        """

                Post-process results: remove duplicates, sort by confidence

        for i, detection in enumerate(detections):        """

            print(f"\nCard {i+1}:")        if not results:

            print(f"  Champion: {detection['champion']}")            return []

            print(f"  CLIP Confidence: {detection['champion_confidence']:.3f}")        

            print(f"  YOLO Confidence: {detection['confidence']:.3f}")        # Remove duplicates using Non-Maximum Suppression

            print(f"  BBox: {detection['bbox']}")        filtered_results = self._non_maximum_suppression(results, iou_threshold=0.4)

                    

            # Show top predictions        # Sort by CLIP confidence (most confident first)

            if "predictions" in detection:        filtered_results.sort(key=lambda x: x["clip_confidence"], reverse=True)

                print(f"  Top predictions:")        

                for j, pred in enumerate(detection["predictions"][:3]):        # Return up to expected count

                    print(f"    {j+1}. {pred['champion']}: {pred['confidence']:.3f}")        return filtered_results[:expected_count]

            

        # Create visualization    def _non_maximum_suppression(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:

        vis_image = detector.visualize_results(image, detections)        """Remove overlapping detections"""

        output_path = "yolo_clip_detection_result.jpg"        if not detections:

        cv2.imwrite(output_path, vis_image)            return []

        print(f"\n💾 Visualization saved to: {output_path}")        

                # Sort by CLIP confidence

        # Save results        sorted_detections = sorted(detections, key=lambda x: x["clip_confidence"], reverse=True)

        detector.save_results(detections, "detection_results.json", include_crops=True)        

                keep = []

        # Run ban/pick analysis        while sorted_detections:

        analysis = detector.analyze_banpick(image)            # Keep highest confidence detection

        print(f"\n📊 Ban/Pick Analysis:")            current = sorted_detections.pop(0)

        print(f"Champions detected: {', '.join(analysis['champions_detected'])}")            keep.append(current)

        print(f"Average confidence: {analysis['average_confidence']:.3f}")            

                    # Remove overlapping detections

    else:            remaining = []

        print(f"❌ Test image not found: {test_image_path}")            for det in sorted_detections:

        print("Pipeline initialized successfully! Ready for detection.")                if self._calculate_iou(current["bounding_box"], det["bounding_box"]) < iou_threshold:

                    remaining.append(det)

            

if __name__ == "__main__":            sorted_detections = remaining

    main()        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_results(self, image_path: str, results: Dict, output_path: str = None) -> str:
        """Create beautiful visualization with champion names and bounding boxes"""
        
        image = cv2.imread(str(image_path))
        
        # Colors for different confidence levels
        colors = {
            "high": (0, 255, 0),    # Green for >0.8
            "medium": (0, 255, 255), # Yellow for 0.6-0.8
            "low": (0, 0, 255)       # Red for <0.6
        }
        
        for i, champion in enumerate(results["champions"]):
            x1, y1, x2, y2 = champion["bounding_box"]
            confidence = champion["clip_confidence"]
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = colors["high"]
            elif confidence > 0.6:
                color = colors["medium"]  
            else:
                color = colors["low"]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Champion name and info
            champion_name = champion["champion_name"]
            label = f"{i+1}. {champion_name}"
            confidence_label = f"CLIP: {confidence:.3f}"
            
            # Text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            conf_size = cv2.getTextSize(confidence_label, font, 0.5, 1)[0]
            
            # Background rectangles
            cv2.rectangle(image, (x1, y1-40), (x1 + max(label_size[0], conf_size[0]) + 10, y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1+5, y1-25), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(image, confidence_label, (x1+5, y1-5), font, 0.5, (0, 0, 0), 1)
        
        # Add summary
        summary = f"YOLO+CLIP: {results['total_detected']} champions detected"
        cv2.putText(image, summary, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, summary, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Save result
        if output_path is None:
            output_path = f"yolo_clip_result_{Path(image_path).stem}.jpg"
        
        cv2.imwrite(output_path, image)
        logger.info(f"🎨 Visualization saved: {output_path}")
        
        return output_path