# STEP-BY-STEP COMMANDS

## IMPORTANT: Always activate virtual environment first

```bash
source venv/bin/activate
```

---

## STEP 1: Generate Training Data (5-10 minutes)

```bash
python3 generate_yolo_dataset.py
```

Expected output:
- Creates `dataset/images/` with training images
- Creates `dataset/labels/` with YOLO labels
- Should generate 500+ images

Verify:
```bash
ls dataset/images/*.jpg | wc -l
ls dataset/labels/*.txt | wc -l
```

---

## STEP 2: Train YOLO Model (30-60 minutes with GPU)

```bash
python3 train_yolo.py
```

Expected output:
- Downloads YOLOv8 base model (first time)
- Trains for 100 epochs
- Saves model to `runs/detect/champion_card_detection/weights/best.pt`
- Shows training progress with loss/precision/recall metrics

What to watch:
- Loss should decrease over time
- Precision and recall should increase
- Training will show progress bar for each epoch

---

## STEP 3: Test Detection (1-2 seconds per image)

You need a real ban-pick screenshot image first.

```bash
# Basic detection
python3 detect_banpick.py path/to/your/banpick_image.jpg

# Save JSON results
python3 detect_banpick.py image.jpg --output results.json

# Save visualization with bounding boxes
python3 detect_banpick.py image.jpg --visualize output_with_boxes.jpg

# Save both
python3 detect_banpick.py image.jpg --output results.json --visualize output.jpg
```

Expected output:
```
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
  ...
```

---

## ALTERNATIVE: Run Everything Automatically

```bash
source venv/bin/activate
chmod +x train_and_setup.sh
./train_and_setup.sh
```

This script will:
1. Check GPU availability
2. Install dependencies if needed
3. Generate training dataset
4. Train YOLO model
5. Show usage examples

---

## Python API Usage

After training is complete, use in your Python code:

```python
from detect_banpick import BanPickDetector

# Initialize detector (loads trained model automatically)
detector = BanPickDetector()

# Detect from image
detections = detector.detect_from_file("banpick_screenshot.jpg")

# Get top 10 champion names
champion_names = detector.get_champion_names(detections, top_n=10)
print(champion_names)
# Output: ['Yasuo', 'Ahri', 'Katarina', 'Lux', 'Jinx', ...]

# Get detailed information
for det in detections:
    print(f"{det['champion_name']}: {det['confidence']:.2f}")
    print(f"  Team: {det['position']['team']}")
    print(f"  Phase: {det['position']['phase']}")

# Save results
detector.save_results(detections, "results.json")
detector.visualize_results("banpick_screenshot.jpg", detections, "output.jpg")
```

---

## Troubleshooting

### If training is slow

Check GPU usage:
```bash
nvidia-smi
```

If GPU is not being used, ensure CUDA is properly installed.

### If detection accuracy is low

1. Generate more training data:
   Edit `generate_yolo_dataset.py` and increase `num_samples`

2. Train longer:
   Edit `train_yolo.py` and increase `epochs` to 200

3. Check training metrics:
   Look at `runs/detect/champion_card_detection/results.png`

### If model not found

Training may have failed. Check:
```bash
ls runs/detect/champion_card_detection/weights/
```

Should show `best.pt` and `last.pt`

---

## Quick Summary

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Generate data
python3 generate_yolo_dataset.py

# 3. Train model
python3 train_yolo.py

# 4. Detect champions
python3 detect_banpick.py your_image.jpg
```

Total time: ~1 hour (mostly training)
Result: Working champion detection system
