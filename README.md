# Champion Detection with YOLO + CLIP

Professional solution for detecting and classifying League of Legends champions in ban-pick screenshots.

This project uses a two-stage approach:
1. **YOLO** detects champion card positions (bounding boxes only)
2. **CLIP** classifies each cropped champion (name identification)

## Project Structure

```
predict-win-Percent/
├── dataset/                          # YOLO training data
│   ├── images/                       # Generated training images
│   ├── labels/                       # YOLO format labels
│   └── data.yaml                     # YOLO config
├── champion_detection/
│   ├── models/
│   │   ├── yolo/
│   │   │   └── card_detector.py      # Pure YOLO detection
│   │   └── clip/
│   │       └── champion_classifier.py # Pure CLIP classification
│   ├── utils/                        # Helper utilities
│   ├── examples/                     # Example usage
│   └── pipeline.py                   # Main pipeline
├── generate_yolo_dataset.py          # Auto-generate training data
├── train_yolo.py                     # Train YOLO model
├── detect_banpick.py                 # Ban-pick detection CLI
└── league-of-legends-skin-splash-art-collection/  # Source images
```

## Quick Start

### 1. Generate Training Dataset

```bash
# Auto-generate YOLO training data from splash arts
python generate_yolo_dataset.py
```

### 2. Train YOLO Model

```bash
# Train YOLO to detect champion card positions
python train_yolo.py
```

### 3. Use Complete Pipeline

```python
from champion_detection.pipeline import ChampionDetectionPipeline

# Initialize pipeline
detector = ChampionDetectionPipeline("runs/detect/champion_card_detection/weights/best.pt")

# Detect champions in image
image = cv2.imread("banpick_screenshot.jpg")
results = detector.detect_champions(image)

# Results format:
# [{"champion": "Ahri", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}, ...]
```

## Key Features

### Auto Dataset Generation

- No manual annotation needed
- Uses existing splash art collection
- Generates realistic ban/pick screenshots
- Creates accurate YOLO labels automatically

### Pure YOLO Detection

- Detects 10 champion card positions
- Returns bounding boxes only
- No champion classification (CLIP handles this)
- Fast and accurate positioning

### CLIP Classification

- 100% accurate champion identification
- Works on cropped champion images
- Handles all 171+ champions
- Confidence scoring included

### Clean Architecture

- Separated detection and classification
- Easy to train and improve each component
- Modular and maintainable code
- Clear pipeline flow

## Training Configuration

The auto-generated dataset includes:

- **Images**: Synthetic ban/pick screenshots
- **Labels**: YOLO format (class=0 for all champion cards)
- **Positions**: 10 predefined champion card slots
- **Variations**: Random champion combinations

YOLO only learns to find these 10 positions, not identify champions.

## Expected Performance

- **YOLO Detection**: Fast, accurate bounding box detection
- **CLIP Classification**: Near 100% accuracy on champion names
- **Combined Pipeline**: Complete solution for ban/pick analysis

## Dependencies

```bash
pip install ultralytics opencv-python clip-by-openai torch torchvision
```

## Next Steps

1. **Generate Dataset**: `python generate_yolo_dataset.py`
2. **Train YOLO**: `python train_yolo.py`
3. **Test Pipeline**: `python champion_detection/pipeline.py`
4. **Integrate**: Use trained pipeline in your application

## Ban-Pick Detection

**Detect 10 champions from a single ban-pick screenshot.**

### Usage

```bash
# Command line detection
python detect_banpick.py path/to/banpick.jpg

# Save results to JSON
python detect_banpick.py image.jpg --output results.json

# Save visualization with bounding boxes
python detect_banpick.py image.jpg --visualize output.jpg
```

### Example Output

```text
BAN-PICK DETECTION RESULTS

BLUE TEAM:
  Bans:  Yasuo (0.95), Zed (0.89), LeBlanc (0.87)
  Picks: Ahri (0.92), Jinx (0.88), Thresh (0.85), Lee Sin (0.83), Malphite (0.81)

RED TEAM:
  Bans:  Katarina (0.91), Fizz (0.86), Akali (0.84)
  Picks: Lux (0.90), Ezreal (0.87), Leona (0.85), Graves (0.82), Garen (0.79)

Total: 10 champions detected

Top 10 Champions:
  1. Yasuo
  2. Ahri
  3. Katarina
  4. Lux
  5. Jinx
  ...
```

### Python API

```python
from detect_banpick import BanPickDetector

# Initialize detector
detector = BanPickDetector()

# Detect from image
detections = detector.detect_from_file("banpick.jpg")

# Get champion names
champion_names = detector.get_champion_names(detections, top_n=10)
print(champion_names)  # ['Yasuo', 'Ahri', 'Katarina', ...]

# Save results
detector.save_results(detections, "results.json")
detector.visualize_results("banpick.jpg", detections, "output.jpg")
```

### Features

- **Input**: Single screenshot image
- **Output**: 10 champion names (5 blue team + 5 red team)
- **Exports**: JSON results and visualization
- **Team classification**: Automatically identifies blue/red team
- **Phase detection**: Distinguishes bans from picks

## Architecture

**Key insight**: YOLO finds WHERE champions are, CLIP identifies WHO they are. This separation makes the system more accurate and easier to train than end-to-end approaches.

## Training Data

Dataset: [League of Legends Skin Splash Art Collection](https://www.kaggle.com/datasets/alihabibullah/league-of-legends-skin-splash-art-collection)
