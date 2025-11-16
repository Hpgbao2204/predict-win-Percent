"""
Main Champion Detection System
Combines CLIP and OpenAI Vision API for robust champion detection
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import json
from datetime import datetime

from models.clip_detector import CLIPChampionDetector
from utils.openai_vision import OpenAIVisionDetector
from utils.image_utils import setup_logging, create_results_directory
from config.settings import *

logger = logging.getLogger(__name__)

class ChampionDetectionSystem:
    """
    Main system combining multiple detection methods
    """
    
    def __init__(self, use_clip: bool = True, use_openai: bool = False):
        """
        Initialize detection system
        
        Args:
            use_clip: Enable CLIP-based detection
            use_openai: Enable OpenAI Vision API detection
        """
        setup_logging()
        create_results_directory()
        
        logger.info("Initializing Champion Detection System...")
        
        self.clip_detector = None
        self.openai_detector = None
        
        # Initialize CLIP detector
        if use_clip:
            try:
                logger.info("Loading CLIP detector...")
                self.clip_detector = CLIPChampionDetector()
                logger.info("CLIP detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP detector: {str(e)}")
        
        # Initialize OpenAI detector
        if use_openai:
            try:
                logger.info("Loading OpenAI Vision detector...")
                self.openai_detector = OpenAIVisionDetector()
                logger.info("OpenAI Vision detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OpenAI Vision detector: {str(e)}")
        
        if not self.clip_detector and not self.openai_detector:
            raise RuntimeError("No detection models could be loaded")
    
    def detect_champion(self, 
                       image_path: Union[str, Path], 
                       method: str = "auto",
                       save_results: bool = True) -> Dict:
        """
        Detect champion from image using specified method
        
        Args:
            image_path: Path to input image
            method: Detection method ("clip", "openai", "auto", "ensemble")
            save_results: Whether to save results to file
            
        Returns:
            Detection results dictionary
        """
        image_path = Path(image_path)
        start_time = time.time()
        
        logger.info(f"Detecting champion in: {image_path}")
        logger.info(f"Detection method: {method}")
        
        results = {
            "input_image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "results": {},
            "final_prediction": None,
            "processing_time": 0
        }
        
        # CLIP detection
        if method in ["clip", "auto", "ensemble"] and self.clip_detector:
            logger.info("Running CLIP detection...")
            try:
                clip_results = self.clip_detector.detect_champion(image_path, top_k=5)
                results["results"]["clip"] = clip_results
                logger.info(f"CLIP detection completed: {len(clip_results)} results")
            except Exception as e:
                logger.error(f"CLIP detection failed: {str(e)}")
                results["results"]["clip"] = []
        
        # OpenAI Vision detection
        if method in ["openai", "auto", "ensemble"] and self.openai_detector:
            logger.info("Running OpenAI Vision detection...")
            try:
                openai_result = self.openai_detector.detect_champion(image_path)
                results["results"]["openai"] = openai_result
                logger.info("OpenAI Vision detection completed")
            except Exception as e:
                logger.error(f"OpenAI Vision detection failed: {str(e)}")
                results["results"]["openai"] = None
        
        # Determine final prediction
        results["final_prediction"] = self._determine_final_prediction(results["results"], method)
        results["processing_time"] = time.time() - start_time
        
        logger.info(f"Detection completed in {results['processing_time']:.2f} seconds")
        if results["final_prediction"]:
            logger.info(f"Final prediction: {results['final_prediction']['champion_name']} "
                       f"(confidence: {results['final_prediction']['confidence']:.3f})")
        
        # Save results
        if save_results:
            self._save_results(results)
        
        return results
    
    def _determine_final_prediction(self, detection_results: Dict, method: str) -> Optional[Dict]:
        """
        Determine final prediction from multiple detection results
        
        Args:
            detection_results: Results from different detection methods
            method: Detection method used
            
        Returns:
            Final prediction dictionary
        """
        if method == "clip" and "clip" in detection_results and detection_results["clip"]:
            return detection_results["clip"][0]
        
        if method == "openai" and "openai" in detection_results:
            return detection_results["openai"]
        
        if method in ["auto", "ensemble"]:
            # Prioritize OpenAI if available and confident
            if ("openai" in detection_results and 
                detection_results["openai"] and 
                detection_results["openai"]["confidence"] > 0.8):
                return detection_results["openai"]
            
            # Otherwise use CLIP if available
            if ("clip" in detection_results and 
                detection_results["clip"] and 
                detection_results["clip"][0]["confidence"] > SIMILARITY_THRESHOLD):
                return detection_results["clip"][0]
            
            # Fallback to best available result
            if "openai" in detection_results and detection_results["openai"]:
                return detection_results["openai"]
            elif "clip" in detection_results and detection_results["clip"]:
                return detection_results["clip"][0]
        
        return None
    
    def _save_results(self, results: Dict):
        """Save detection results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def batch_detect(self, 
                    image_directory: Union[str, Path],
                    method: str = "auto",
                    pattern: str = "*",
                    save_batch_results: bool = True) -> List[Dict]:
        """
        Detect champions for all images in a directory
        
        Args:
            image_directory: Directory containing images
            method: Detection method to use
            pattern: File pattern to match (e.g., "*.jpg")
            save_batch_results: Whether to save batch results
            
        Returns:
            List of detection results
        """
        image_dir = Path(image_directory)
        if not image_dir.exists():
            logger.error(f"Directory not found: {image_dir}")
            return []
        
        # Find all image files
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(image_dir.glob(f"{pattern}{ext}"))
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        # Process each image
        batch_results = []
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            result = self.detect_champion(image_file, method=method, save_results=False)
            batch_results.append(result)
        
        # Save batch results
        if save_batch_results:
            self._save_batch_results(batch_results, image_dir, method)
        
        return batch_results
    
    def _save_batch_results(self, batch_results: List[Dict], source_dir: Path, method: str):
        """Save batch detection results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_detection_{source_dir.name}_{method}_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            batch_summary = {
                "source_directory": str(source_dir),
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "total_images": len(batch_results),
                "successful_detections": sum(1 for r in batch_results if r["final_prediction"]),
                "results": batch_results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {str(e)}")
    
    def get_system_info(self) -> Dict:
        """Get information about the detection system"""
        info = {
            "available_methods": [],
            "clip_info": None,
            "openai_available": self.openai_detector is not None,
            "settings": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "top_k_results": TOP_K_RESULTS,
                "max_image_size": MAX_IMAGE_SIZE
            }
        }
        
        if self.clip_detector:
            info["available_methods"].append("clip")
            info["clip_info"] = self.clip_detector.get_model_info()
        
        if self.openai_detector:
            info["available_methods"].append("openai")
        
        return info