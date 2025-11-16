"""
Data loader for champion database
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from config.settings import SKINS_JSON_FILE

logger = logging.getLogger(__name__)

class ChampionDataLoader:
    """Load and manage champion skin data"""
    
    def __init__(self):
        """Initialize data loader"""
        self.champions_data = self._load_champions_data()
    
    def _load_champions_data(self) -> Dict:
        """Load champions data from JSON file"""
        try:
            # Get the absolute path for JSON file (relative to champion_detection directory)
            current_dir = Path(__file__).parent.parent  # champion_detection directory
            json_file_path = current_dir / SKINS_JSON_FILE
            
            if not json_file_path.exists():
                logger.error(f"Champion data file not found: {json_file_path}")
                return {}
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert relative paths in JSON to absolute paths for processing
            for champion_key, champion_data in data.items():
                if isinstance(champion_data, dict) and 'skins' in champion_data:
                    # Convert relative paths to absolute paths
                    updated_skins = []
                    for skin_path in champion_data['skins']:
                        if skin_path.startswith('../'):
                            # Convert relative path to absolute path
                            abs_path = current_dir / skin_path
                            updated_skins.append(str(abs_path.resolve()))
                        else:
                            updated_skins.append(skin_path)
                    champion_data['skins'] = updated_skins
            
            logger.info(f"Loaded {len(data)} champions from database")
            return data
            
        except Exception as e:
            logger.error(f"Error loading champion data: {e}")
            return {}
    
    def get_champion_skins(self, champion_name: str) -> List[str]:
        """Get list of skin paths for a champion"""
        for champion_key, champion_data in self.champions_data.items():
            if champion_data.get("original_name", "").lower() == champion_name.lower():
                return champion_data.get("skins", [])
        return []
    
    def get_all_champions(self) -> List[str]:
        """Get list of all champion names"""
        return [data["original_name"] for data in self.champions_data.values()]