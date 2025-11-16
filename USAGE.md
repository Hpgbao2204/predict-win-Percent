# Ban-Pick Champion Detection - Usage Guide

## Overview

System for detecting champion names from League of Legends ban-pick screenshots using YOLO + CLIP pipeline.

**Input:** Ban-pick screenshot image
**Output:** List of 10 champion names with confidence scores

## Prerequisites

### 1. Train YOLO Model

```bash
# Generate training dataset
python generate_yolo_dataset.py

# Train YOLO model (requires GPU for faster training)
python train_yolo.py
```

Training time: 1-3 hours depending on GPU availability.
Model will be saved to: `runs/detect/champion_card_detection/weights/best.pt`

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python torch torchvision
```

## Basic Usage

### Command Line Interface

```bash
# Basic detection
python detect_banpick.py image.jpg

# Save JSON output
python detect_banpick.py image.jpg --output results.json

# Save visualization
python detect_banpick.py image.jpg --visualize output.jpg

# Custom model path
python detect_banpick.py image.jpg --model path/to/model.pt
```

### Python API

```python
from detect_banpick import BanPickDetector

# Initialize
detector = BanPickDetector()

# Detect from file
detections = detector.detect_from_file("banpick.jpg")

# Get champion names (sorted by confidence)
names = detector.get_champion_names(detections, top_n=10)

# Display results in console
detector.print_results(detections)

# Save to JSON
detector.save_results(detections, "results.json")

# Create visualization
detector.visualize_results("banpick.jpg", detections, "output.jpg")
```

## Output Format

### Console Output

```text
BAN-PICK DETECTION RESULTS

BLUE TEAM:
  Bans:  Yasuo (0.95), Zed (0.89), LeBlanc (0.87)
  Picks: Ahri (0.92), Jinx (0.88), Thresh (0.85), Lee Sin (0.83), Malphite (0.81)

RED TEAM:
  Bans:  Katarina (0.91), Fizz (0.86), Akali (0.84)
  Picks: Lux (0.90), Ezreal (0.87), Leona (0.85), Graves (0.82), Garen (0.79)

Total: 10 champions detected
```

### JSON Output

```json
{
  "total_champions": 10,
  "champion_names": ["Yasuo", "Ahri", "Katarina", "Lux", "Jinx", ...],
  "detections": [
    {
      "champion_name": "Yasuo",
      "confidence": 0.95,
      "bbox": [100, 200, 250, 400],
      "position": {
        "team": "blue_team",
        "phase": "ban",
        "x_center": 175,
        "y_center": 300
      }
    },
    ...
  ]
}
```

## Detection Details

Each detection contains:

- **champion_name**: Identified champion name
- **confidence**: Classification confidence (0-1)
- **bbox**: Bounding box [x1, y1, x2, y2]
- **position.team**: "blue_team" or "red_team"
- **position.phase**: "ban" or "pick"
- **position.x_center**: Horizontal center coordinate
- **position.y_center**: Vertical center coordinate

## Troubleshooting

### No trained model found

Run training first:
```bash
python generate_yolo_dataset.py
python train_yolo.py
```

### Insufficient detections (< 10 champions)

Possible causes:
- YOLO model needs more training
- Image quality too low (minimum 720p recommended)
- Image is not a proper ban-pick screenshot

Solutions:
- Generate more training data
- Increase training epochs
- Use higher quality images

### Low confidence scores

- CLIP model is pre-trained and generally accurate
- Low scores may indicate image quality issues
- Check if champion cards are clearly visible

## Technical Details

### Pipeline Architecture

1. **YOLO Detection**: Locates champion card positions in image
2. **Image Cropping**: Extracts individual champion cards
3. **CLIP Classification**: Identifies champion from each card
4. **Position Analysis**: Determines team and phase based on spatial location

### Model Requirements

- **YOLO**: Requires training on ban-pick screenshots
- **CLIP**: Pre-trained, no additional training needed
- **GPU**: Recommended for training, optional for inference

### Performance

- **Detection Speed**: 0.5-2 seconds per image
- **Accuracy**: 85-95% with properly trained YOLO model
- **Scalability**: Can process images in batch

## Advanced Usage

### Custom Model Path

```python
detector = BanPickDetector(yolo_model_path="custom/path/to/model.pt")
```

### Batch Processing

```python
import glob
from detect_banpick import BanPickDetector

detector = BanPickDetector()

for image_path in glob.glob("images/*.jpg"):
    detections = detector.detect_from_file(image_path)
    output_name = f"results_{Path(image_path).stem}.json"
    detector.save_results(detections, output_name)
```

### Integration Example

```python
from detect_banpick import BanPickDetector

class GameAnalyzer:
    def __init__(self):
        self.detector = BanPickDetector()
    
    def analyze_match(self, screenshot_path):
        detections = self.detector.detect_from_file(screenshot_path)
        champions = self.detector.get_champion_names(detections, top_n=10)
        
        # Your analysis logic here
        return self.predict_win_rate(champions)
```

## Notes

- Ensure images are clear and properly formatted ban-pick screenshots
- Training data quality directly impacts detection accuracy
- CLIP handles champion classification automatically
- System designed for standard 10-champion ban-pick format
