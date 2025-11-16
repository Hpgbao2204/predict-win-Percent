"""
Configuration settings for Champion Detection System
"""

import os
from pathlib import Path

# Base paths - using relative paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = Path("../league-of-legends-skin-splash-art-collection")
SKINS_DIR = DATA_DIR / "skins"

# JSON files paths - using relative paths
SKINS_DETAILED_JSON = Path("../data-scrapping/skins_detailed.json")
SKINS_SIMPLE_JSON = Path("../data-scrapping/skins_simple.json")
SKINS_JSON_FILE = SKINS_DETAILED_JSON  # Default to detailed version

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
CLIP_MODEL_NAME = "ViT-B/32"  # or "ViT-L/14" for better accuracy
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 5

# Detection Configuration
MAX_IMAGE_SIZE = (512, 512)
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# Output Configuration
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"