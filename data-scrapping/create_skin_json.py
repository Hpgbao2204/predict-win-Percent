#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create JSON file containing information about all skin splash art
"""

import os
import json
from pathlib import Path

def scan_skins_directory(base_path):
    """
    Scan skins directory and create dictionary containing information about all skins
    
    Args:
        base_path (str): Path to directory containing champion folders
        
    Returns:
        dict: Dictionary containing skin information in format:
              {
                  "champion_name": {
                      "skins": ["skin1.jpg", "skin2.jpg", ...],
                      "total_skins": number_of_skins
                  }
              }
    """
    skins_data = {}
    skins_path = Path(base_path) / "skins"
    
    if not skins_path.exists():
        print(f"Directory not found: {skins_path}")
        return skins_data
    
    # Scan through all champion folders
    for champion_folder in sorted(skins_path.iterdir()):
        if champion_folder.is_dir():
            champion_name = champion_folder.name.lower().replace(" ", "_").replace("-", "_")
            print(f"Processing: {champion_folder.name}")
            
            # Get all image files in champion folder
            skin_files = []
            for file_path in champion_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Create relative path from project root
                    relative_path = str(file_path.relative_to(Path(base_path)))
                    skin_files.append(relative_path)
            
            # Add to dictionary
            if skin_files:
                skins_data[champion_name] = {
                    "original_name": champion_folder.name,
                    "skins": sorted(skin_files),
                    "total_skins": len(skin_files)
                }
    
    return skins_data

def create_simple_mapping(skins_data):
    """
    Create simple mapping according to required format
    
    Args:
        skins_data (dict): Dictionary containing detailed skin information
        
    Returns:
        dict: Simple dictionary in format: {"champion": "path_to_default_skin"}
    """
    simple_mapping = {}
    
    for champion, data in skins_data.items():
        if data["skins"]:
            # Find default skin (usually the skin with same name as champion)
            default_skin = None
            original_name = data["original_name"]
            
            # Find skin with exact champion name
            for skin_path in data["skins"]:
                skin_filename = os.path.basename(skin_path)
                if skin_filename.lower() == f"{original_name.lower()}.jpg":
                    default_skin = skin_path
                    break
            
            # If not found, take the first skin
            if not default_skin:
                default_skin = data["skins"][0]
            
            simple_mapping[champion] = default_skin
    
    return simple_mapping

def main():
    """Main function"""
    base_path = "../league-of-legends-skin-splash-art-collection"
    
    print("Starting to scan skins directory...")
    skins_data = scan_skins_directory(base_path)
    
    if not skins_data:
        print("No skin data found!")
        return
    
    # Create detailed JSON file
    detailed_output_file = "skins_detailed.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(skins_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {detailed_output_file} with {len(skins_data)} champions")
    
    # Create simple JSON file as requested
    simple_mapping = create_simple_mapping(skins_data)
    simple_output_file = "skins_simple.json"
    
    with open(simple_output_file, 'w', encoding='utf-8') as f:
        json.dump(simple_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {simple_output_file} with {len(simple_mapping)} champions")
    
    # Display some examples
    print("\n📊 Statistics:")
    total_skins = sum(data["total_skins"] for data in skins_data.values())
    print(f"- Total champions: {len(skins_data)}")
    print(f"- Total skins: {total_skins}")
    
    print("\n🎨 Example champions:")
    for i, (champion, data) in enumerate(list(skins_data.items())[:5]):
        print(f"- {data['original_name']}: {data['total_skins']} skins")
    
    print(f"\n📁 JSON files created in current directory:")
    print(f"- {detailed_output_file}: Detailed information of all skins")
    print(f"- {simple_output_file}: Simple mapping champion -> default skin")

if __name__ == "__main__":
    main()