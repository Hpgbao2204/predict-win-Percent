#!/bin/bash
# Quick Start Script - Complete Training and Detection Setup
# Run this script to train the model and test detection

set -e  # Exit on error

echo "=========================================="
echo "Ban-Pick Champion Detection - Quick Start"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo ""

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "Installing required packages..."
    pip install ultralytics opencv-python torch torchvision
fi
echo "Dependencies OK"
echo ""

# Step 2: Generate training dataset
echo "Step 2: Generating training dataset..."
if [ ! -d "dataset/images" ] || [ -z "$(ls -A dataset/images 2>/dev/null)" ]; then
    echo "Creating synthetic ban-pick images for training..."
    python3 generate_yolo_dataset.py
    echo "Dataset generation complete"
else
    echo "Dataset already exists ($(ls dataset/images/*.jpg 2>/dev/null | wc -l) images)"
    read -p "Regenerate dataset? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 generate_yolo_dataset.py
    fi
fi
echo ""

# Step 3: Train YOLO model
echo "Step 3: Training YOLO model..."
if [ ! -f "runs/detect/champion_card_detection/weights/best.pt" ]; then
    echo "Starting YOLO training (this will take 30-60 minutes with GPU)..."
    echo "Training parameters:"
    echo "  - Epochs: 100"
    echo "  - Batch size: 16"
    echo "  - Image size: 640"
    echo ""
    python3 train_yolo.py
    echo "Training complete!"
else
    echo "Trained model already exists"
    read -p "Retrain model? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 train_yolo.py
    fi
fi
echo ""

# Step 4: Ready to use
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your system is now ready to detect champions from ban-pick screenshots."
echo ""
echo "Usage examples:"
echo ""
echo "1. Detect from image:"
echo "   python3 detect_banpick.py your_image.jpg"
echo ""
echo "2. Save results to JSON:"
echo "   python3 detect_banpick.py image.jpg --output results.json"
echo ""
echo "3. Save visualization:"
echo "   python3 detect_banpick.py image.jpg --visualize output.jpg"
echo ""
echo "4. Python API:"
echo "   from detect_banpick import BanPickDetector"
echo "   detector = BanPickDetector()"
echo "   detections = detector.detect_from_file('image.jpg')"
echo "   names = detector.get_champion_names(detections, top_n=10)"
echo ""
echo "Model location: runs/detect/champion_card_detection/weights/best.pt"
echo "=========================================="
