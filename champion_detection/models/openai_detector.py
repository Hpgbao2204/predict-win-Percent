"""
OpenAI Vision API based champion detection
"""

import base64
import io
from PIL import Image
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json
import re

from openai import OpenAI
from config.settings import *
from utils.image_utils import ImageProcessor, DataLoader

logger = logging.getLogger(__name__)

class OpenAIVisionDetector:
    """OpenAI Vision API based champion detection system"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.data_loader = DataLoader()
        self.image_processor = ImageProcessor()
        
        # Create champion names list for prompt
        self.champion_names = self.data_loader.get_champion_names()
        
    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> Optional[str]:
        """
        Encode image to base64 for OpenAI API
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Load image
            image_array = self.image_processor.load_image(image_path)
            if image_array is None:
                return None
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array)
            
            # Resize if too large (OpenAI has size limits)
            max_size = 512
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None
    
    def _create_detection_prompt(self) -> str:
        """Create prompt for champion detection"""
        champion_list = ", ".join(self.champion_names[:50])  # Limit to avoid token limits
        
        prompt = f"""
You are an expert at identifying League of Legends champions from images. 

Analyze the provided image and identify which League of Legends champion is shown. The image might be:
- A splash art of a champion
- A skin splash art
- In-game screenshot
- Fan art or concept art

Please provide your analysis in the following JSON format:
{{
    "champion_name": "exact champion name",
    "confidence": 0.95,
    "reasoning": "detailed explanation of visual features that led to identification",
    "alternative_matches": ["other possible champions if uncertain"],
    "skin_name": "skin name if identifiable, otherwise null"
}}

Available champions include (partial list): {champion_list}

Focus on distinctive visual features like:
- Weapon type and appearance
- Character silhouette and pose
- Color scheme and clothing
- Unique abilities or visual effects
- Facial features and hair style
- Background elements

Be precise with champion names and provide confidence score between 0.0 and 1.0.
"""
        return prompt
    
    def detect_champion(self, image_path: Union[str, Path]) -> Optional[Dict]:
        """
        Detect champion using OpenAI Vision API
        
        Args:
            image_path: Path to input image
            
        Returns:
            Detection result dictionary
        """
        try:
            # Encode image
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("Failed to encode image")
                return None
            
            # Create prompt
            prompt = self._create_detection_prompt()
            
            # Call OpenAI Vision API
            logger.info("Calling OpenAI Vision API...")
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            logger.info(f"OpenAI Response: {response_text}")
            
            # Extract JSON from response
            result = self._parse_response(response_text)
            
            if result:
                # Add metadata
                result["detection_method"] = "openai_vision"
                result["model_used"] = "gpt-4-vision-preview"
                result["input_image"] = str(image_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in OpenAI Vision detection: {str(e)}")
            return None
    
    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse OpenAI response to extract JSON
        
        Args:
            response_text: Raw response from OpenAI
            
        Returns:
            Parsed result dictionary
        """
        try:
            # Try to find JSON in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    result = json.loads(match)
                    # Validate required fields
                    if "champion_name" in result and "confidence" in result:
                        return result
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, try to extract manually
            return self._manual_parse(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}")
            return None
    
    def _manual_parse(self, text: str) -> Optional[Dict]:
        """
        Manually parse response if JSON extraction fails
        
        Args:
            text: Response text to parse
            
        Returns:
            Parsed result dictionary
        """
        try:
            result = {
                "champion_name": None,
                "confidence": 0.0,
                "reasoning": text,
                "alternative_matches": [],
                "skin_name": None
            }
            
            # Look for champion names in the text
            text_lower = text.lower()
            for champion_name in self.champion_names:
                if champion_name.lower() in text_lower:
                    result["champion_name"] = champion_name
                    result["confidence"] = 0.7  # Default confidence
                    break
            
            return result if result["champion_name"] else None
            
        except Exception as e:
            logger.error(f"Error in manual parsing: {str(e)}")
            return None
    
    def batch_detect(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Detect champions for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of detection results
        """
        results = []
        
        for image_path in image_paths:
            logger.info(f"Processing: {image_path}")
            result = self.detect_champion(image_path)
            
            if result:
                results.append(result)
            else:
                results.append({
                    "input_image": str(image_path),
                    "champion_name": None,
                    "confidence": 0.0,
                    "error": "Detection failed"
                })
        
        return results