"""
CLIP Champion Classification
Pure classification - takes cropped champion images and returns champion names
"""

import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

class CLIPChampionClassifier:
    """
    Pure CLIP classification for champion identification
    
    Responsibilities:
    - Load CLIP model
    - Take cropped champion card images
    - Return champion name predictions
    - Handle confidence scoring
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP classifier
        
        Args:
            model_name: CLIP model variant to use
        """
        logger.info("🎯 Initializing CLIP Champion Classifier")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Champion names for text prompts
        self.champion_names = self._load_champion_names()
        
        # Precompute text embeddings for efficiency
        self.text_embeddings = self._precompute_text_embeddings()
        
        logger.info(f"✅ CLIP classifier ready on {self.device}")
        logger.info(f"📋 Loaded {len(self.champion_names)} champion names")
    
    def _load_champion_names(self) -> List[str]:
        """Load list of all LoL champion names"""
        
        # Standard LoL champion names (171 champions as of 2024)
        champions = [
            "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia", "Annie", "Aphelios",
            "Ashe", "Aurelion Sol", "Aurora", "Azir", "Bard", "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar",
            "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko",
            "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen", "Gnar",
            "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna",
            "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "K'Sante", "Kai'Sa", "Kalista", "Karma", "Karthus",
            "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", "Kog'Maw", "LeBlanc",
            "Lee Sin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai",
            "Master Yi", "Milio", "Miss Fortune", "Mordekaiser", "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko",
            "Nidalee", "Nilah", "Nocturne", "Nunu & Willump", "Olaf", "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke",
            "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven",
            "Rumble", "Ryze", "Samira", "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana",
            "Singed", "Sion", "Sivir", "Skarner", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench",
            "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch",
            "Udyr", "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor",
            "Vladimir", "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick",
            "Yuumi", "Zac", "Zed", "Zeri", "Ziggs", "Zilean", "Zoe", "Zyra"
        ]
        
        return sorted(champions)
    
    def _precompute_text_embeddings(self) -> torch.Tensor:
        """Precompute text embeddings for all champions for efficiency"""
        
        logger.info("🔄 Precomputing text embeddings...")
        
        # Create text prompts
        text_prompts = [f"a photo of {champion}" for champion in self.champion_names]
        
        # Tokenize
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Compute embeddings
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        logger.info("✅ Text embeddings precomputed")
        return text_embeddings
    
    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess cropped image for CLIP
        
        Args:
            crop: Cropped champion image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        
        # Convert BGR to RGB
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
        
        # Convert to PIL Image
        pil_image = Image.fromarray(crop_rgb)
        
        # Apply CLIP preprocessing
        preprocessed = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        return preprocessed
    
    def classify_single_crop(self, crop: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Classify a single cropped champion image
        
        Args:
            crop: Cropped champion image (BGR format)
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions: [{"champion": str, "confidence": float}, ...]
        """
        
        try:
            # Preprocess image
            image_tensor = self._preprocess_crop(crop)
            
            # Compute image embedding
            with torch.no_grad():
                image_embedding = self.model.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)
            
            # Get top-k predictions
            top_similarities, top_indices = similarities.topk(top_k)
            
            # Format results
            predictions = []
            for sim, idx in zip(top_similarities, top_indices):
                champion = self.champion_names[idx.item()]
                confidence = sim.item()
                
                predictions.append({
                    "champion": champion,
                    "confidence": confidence
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return []
    
    def classify_multiple_crops(self, crops: List[np.ndarray], top_k: int = 3) -> List[List[Dict]]:
        """
        Classify multiple cropped images
        
        Args:
            crops: List of cropped champion images
            top_k: Number of top predictions per crop
            
        Returns:
            List of prediction lists for each crop
        """
        
        results = []
        
        for i, crop in enumerate(crops):
            logger.info(f"🔍 Classifying crop {i+1}/{len(crops)}")
            predictions = self.classify_single_crop(crop, top_k)
            results.append(predictions)
        
        return results
    
    def classify_detections(self, detections: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Classify champion crops from YOLO detections
        
        Args:
            detections: Detection results with "crop" field from YOLO detector
            top_k: Number of top predictions per detection
            
        Returns:
            Enhanced detections with classification results
        """
        
        classified_detections = []
        
        for detection in detections:
            if "crop" not in detection:
                logger.warning("Detection missing crop - skipping classification")
                continue
            
            # Classify the crop
            predictions = self.classify_single_crop(detection["crop"], top_k)
            
            # Add classification to detection
            enhanced_detection = detection.copy()
            enhanced_detection["predictions"] = predictions
            
            # Add best prediction for convenience
            if predictions:
                enhanced_detection["champion"] = predictions[0]["champion"]
                enhanced_detection["champion_confidence"] = predictions[0]["confidence"]
            else:
                enhanced_detection["champion"] = "Unknown"
                enhanced_detection["champion_confidence"] = 0.0
            
            classified_detections.append(enhanced_detection)
        
        logger.info(f"🎯 Classified {len(classified_detections)} champion crops")
        return classified_detections
    
    def get_champion_summary(self, classified_detections: List[Dict]) -> Dict:
        """
        Get summary of all detected champions
        
        Args:
            classified_detections: Results from classify_detections()
            
        Returns:
            Summary with champion counts and confidence scores
        """
        
        summary = {
            "total_detections": len(classified_detections),
            "champions": {},
            "average_confidence": 0.0,
            "detected_champions": []
        }
        
        total_confidence = 0.0
        
        for detection in classified_detections:
            champion = detection.get("champion", "Unknown")
            confidence = detection.get("champion_confidence", 0.0)
            
            # Count champions
            if champion not in summary["champions"]:
                summary["champions"][champion] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "confidences": []
                }
            
            summary["champions"][champion]["count"] += 1
            summary["champions"][champion]["confidences"].append(confidence)
            
            # Add to detected list
            summary["detected_champions"].append(champion)
            
            total_confidence += confidence
        
        # Calculate averages
        if classified_detections:
            summary["average_confidence"] = total_confidence / len(classified_detections)
        
        for champ_data in summary["champions"].values():
            champ_data["avg_confidence"] = sum(champ_data["confidences"]) / len(champ_data["confidences"])
            del champ_data["confidences"]  # Clean up
        
        return summary


def main():
    """Test CLIP classifier"""
    
    # Initialize classifier
    classifier = CLIPChampionClassifier()
    
    # Test with a sample crop (if available)
    test_crop_path = "../test_images/champion_crop.jpg"
    
    if Path(test_crop_path).exists():
        # Load test crop
        crop = cv2.imread(test_crop_path)
        
        # Classify
        predictions = classifier.classify_single_crop(crop, top_k=5)
        
        print("Champion Classification Results:")
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred['champion']}: {pred['confidence']:.3f}")
    else:
        print(f"Test crop not found: {test_crop_path}")
        print("CLIP classifier initialized successfully!")
        print(f"Ready to classify {len(classifier.champion_names)} champions")


if __name__ == "__main__":
    main()