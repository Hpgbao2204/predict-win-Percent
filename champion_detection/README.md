# League of Legends Champion Detection System

Detect LoL champions from images using CLIP and OpenAI Vision API.

## Quick Start

1. **Install dependencies:**
```bash
cd champion_detection
pip install -r requirements.txt
```

2. **Test with existing skins:**
```bash
python example.py
```

3. **Detect from image:**
```bash
python cli.py detect image.jpg --method clip
```

## Usage

### CLI Commands
```bash
# Single image detection
python cli.py detect image.jpg --method clip

# Batch detection  
python cli.py batch /path/to/images

# System info
python cli.py info
```

### Python API
```python
from champion_detector import ChampionDetectionSystem

detector = ChampionDetectionSystem(use_clip=True)
result = detector.detect_champion("image.jpg")
print(f"Champion: {result['final_prediction']['champion_name']}")
```

## Methods

- **CLIP**: Fast similarity matching with splash art database
- **OpenAI Vision**: GPT-4V analysis (requires API key)
- **Auto**: Combines both methods for best accuracy

## Configuration

Edit `config/settings.py` for custom settings:
```python
CLIP_MODEL_NAME = "ViT-B/32"
SIMILARITY_THRESHOLD = 0.7
MAX_IMAGE_SIZE = (512, 512)
```

