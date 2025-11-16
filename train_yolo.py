#!/usr/bin/env python3
"""
Train YOLO Model for Champion Card Detection

This script trains a YOLO model to detect champion card positions in ban/pick screenshots.
The trained model will be used with CLIP for complete champion detection.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_yolo_model(
    data_yaml: str = "dataset/data.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = "runs/detect",
    name: str = "champion_card_detection"
):
    """
    Train YOLO model for champion card detection
    
    Args:
        data_yaml: Path to data.yaml config file
        model: Base YOLO model to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        project: Project directory for results
        name: Experiment name
    """
    
    print("🎯 Starting YOLO Training for Champion Card Detection")
    print(f"📂 Data config: {data_yaml}")
    print(f"🏗️ Base model: {model}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    
    # Check if dataset exists
    if not Path(data_yaml).exists():
        print(f"❌ Dataset config not found: {data_yaml}")
        print("Please run generate_yolo_dataset.py first to create training data")
        return
    
    # Load YOLO model
    yolo_model = YOLO(model)
    
    # Start training
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        verbose=True,
        save=True,
        plots=True,
        val=True
    )
    
    print("✅ Training completed!")
    print(f"📊 Results saved to: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    
    return results

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        "data_yaml": "dataset/data.yaml",
        "model": "yolov8n.pt",  # Start with nano model for speed
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640
    }
    
    # Check if dataset exists
    dataset_path = Path("dataset/images")
    if not dataset_path.exists() or not list(dataset_path.glob("*.jpg")):
        print("❌ No training images found!")
        print("Please run: python generate_yolo_dataset.py")
        print("This will create training data from splash art collection.")
        sys.exit(1)
    
    # Start training
    results = train_yolo_model(**config)
    
    if results:
        print("\\n🎉 Training Summary:")
        print("1. YOLO model trained to detect champion card positions")
        print("2. Use trained model with CLIP classifier for complete detection")
        print("3. Test pipeline with: python pipeline.py")

if __name__ == "__main__":
    main()