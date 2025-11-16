"""
League of Legends Champion Detection System

A professional system for detecting League of Legends champions from images
using CLIP and OpenAI Vision API.
"""

__version__ = "1.0.0"
__author__ = "Champion Detection Team"

from .champion_detector import ChampionDetectionSystem
from .models.clip_detector import CLIPChampionDetector
from .models.openai_detector import OpenAIVisionDetector

__all__ = [
    'ChampionDetectionSystem',
    'CLIPChampionDetector', 
    'OpenAIVisionDetector'
]