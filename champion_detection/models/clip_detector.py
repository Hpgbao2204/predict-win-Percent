"""
CLIP-based champion detection model
"""

import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

from config.settings import *
from utils.data_loader import ChampionDataLoader
from utils.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class CLIPChampionDetector:
    """CLIP-based champion detection system"""
    
    def __init__(self):
        """Initialize CLIP model for champion detection"""
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"CLIP model loaded successfully on {self.device}")
        
        # Initialize components
        self.model_name = "ViT-B/32"
        self.data_loader = ChampionDataLoader()
        self.image_processor = ImageProcessor()
        
        # Initialize cache and embeddings
        self.embedding_cache_file = Path(CACHE_DIR) / "champion_embeddings.pkl"
        self.champion_embeddings = {}
        
        # Load or create embeddings
        self._load_or_create_embeddings()
    
    def _load_or_create_embeddings(self):
        """Load cached embeddings or create new ones"""
        try:
            # Create cache directory if it doesn't exist
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # Try to load cached embeddings
            if self.embedding_cache_file.exists():
                logger.info("Loading cached embeddings...")
                with open(self.embedding_cache_file, 'rb') as f:
                    self.champion_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.champion_embeddings)} cached embeddings")
            else:
                logger.info("No cached embeddings found. Creating new ones...")
                self._create_champion_embeddings()
                self._save_embeddings()
                
        except Exception as e:
            logger.error(f"Error with embeddings: {str(e)}")
            # Fallback: create new embeddings
            self._create_champion_embeddings()
            self._save_embeddings()
    
    def _load_or_create_embeddings(self):
        """Load cached embeddings or create new ones"""
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.embedding_cache_file.exists():
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    self.champion_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.champion_embeddings)} cached embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {str(e)}")
        
        # Create new embeddings
        self._create_champion_embeddings()
    
    def _create_champion_embeddings(self):
        """Create embeddings for all champion skins"""
        logger.info("Creating champion embeddings...")
        
        champions_data = self.data_loader.champions_data
        if not champions_data:
            logger.error("No champion data available")
            return
        
        total_images = sum(len(data["skins"]) for data in champions_data.values())
        
        with torch.no_grad():
            with tqdm(total=total_images, desc="Processing champion skins") as pbar:
                for champion_key, champion_data in champions_data.items():
                    champion_name = champion_data["original_name"]
                    
                    # Initialize champion embeddings list
                    if champion_key not in self.champion_embeddings:
                        self.champion_embeddings[champion_key] = {
                            "name": champion_name,
                            "embeddings": [],
                            "skin_paths": []
                        }
                    
                    for skin_path in champion_data["skins"]:
                        full_skin_path = PROJECT_ROOT / skin_path
                        
                        if not full_skin_path.exists():
                            logger.warning(f"Skin image not found: {full_skin_path}")
                            pbar.update(1)
                            continue
                        
                        try:
                            # Load and preprocess image
                            image_array = self.image_processor.load_image(full_skin_path)
                            if image_array is None:
                                pbar.update(1)
                                continue
                            
                            # Convert to PIL Image for CLIP preprocessing
                            pil_image = Image.fromarray(image_array)
                            
                            # Preprocess for CLIP
                            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                            
                            # Get image embedding
                            image_features = self.model.encode_image(image_input)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            
                            # Store embedding
                            self.champion_embeddings[champion_key]["embeddings"].append(
                                image_features.cpu().numpy().flatten()
                            )
                            self.champion_embeddings[champion_key]["skin_paths"].append(skin_path)
                            
                        except Exception as e:
                            logger.error(f"Error processing {full_skin_path}: {str(e)}")
                        
                        pbar.update(1)
        
        # Save embeddings to cache
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.champion_embeddings, f)
            logger.info(f"Saved embeddings to cache: {self.embedding_cache_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {str(e)}")
    
    def detect_champion(self, image_path: Union[str, Path], top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Detect champion from input image
        
        Args:
            image_path: Path to input image
            top_k: Number of top matches to return
            
        Returns:
            List of detection results with confidence scores
        """
        try:
            # Load and validate input image
            image_array = self.image_processor.load_image(image_path)
            if image_array is None:
                return []
            
            if not self.image_processor.validate_image(image_array):
                logger.error("Invalid input image")
                return []
            
            # Convert to PIL Image and preprocess
            pil_image = Image.fromarray(image_array)
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get input image embedding
            with torch.no_grad():
                input_features = self.model.encode_image(image_input)
                input_features = input_features / input_features.norm(dim=-1, keepdim=True)
                input_embedding = input_features.cpu().numpy().flatten()
            
            # Compare with all champion embeddings
            results = []
            
            for champion_key, champion_data in self.champion_embeddings.items():
                if not champion_data["embeddings"]:
                    continue
                
                # Calculate similarities with all skins of this champion
                similarities = []
                for embedding in champion_data["embeddings"]:
                    similarity = np.dot(input_embedding, embedding)
                    similarities.append(similarity)
                
                # Use max similarity for this champion
                max_similarity = max(similarities)
                best_skin_idx = similarities.index(max_similarity)
                best_skin_path = champion_data["skin_paths"][best_skin_idx]
                
                results.append({
                    "champion_key": champion_key,
                    "champion_name": champion_data["name"],
                    "confidence": float(max_similarity),
                    "best_matching_skin": best_skin_path,
                    "total_skins_compared": len(similarities)
                })
            
            # Sort by confidence and return top-k
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in champion detection: {str(e)}")
            return []
    
    def detect_with_threshold(self, image_path: Union[str, Path], 
                            threshold: float = SIMILARITY_THRESHOLD) -> Optional[Dict]:
        """
        Detect champion with confidence threshold
        
        Args:
            image_path: Path to input image
            threshold: Minimum confidence threshold
            
        Returns:
            Detection result if above threshold, None otherwise
        """
        results = self.detect_champion(image_path, top_k=1)
        
        if results and results[0]["confidence"] >= threshold:
            return results[0]
        
        return None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "champions_loaded": len(self.champion_embeddings),
            "total_embeddings": sum(len(data["embeddings"]) 
                                  for data in self.champion_embeddings.values()),
            "cache_file": str(self.embedding_cache_file)
        }