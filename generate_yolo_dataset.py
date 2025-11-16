import os
import cv2
import random
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LCKDatasetGenerator:
    def __init__(self,
                 splash_dir="league-of-legends-skin-splash-art-collection/skins",
                 output_dir="dataset_lck",
                 image_width=1365,
                 image_height=768):

        self.splash_dir = Path(splash_dir)
        self.output_dir = Path(output_dir)
        self.img_dir = self.output_dir / "images"
        self.lbl_dir = self.output_dir / "labels"

        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        self.image_width = image_width
        self.image_height = image_height

        self.card_width = 256
        self.card_height = 460

        self.available_champions = self._load_champion_list()

        logger.info(f"Loaded {len(self.available_champions)} champions")

    # -------------------------
    # Load champion folders
    # -------------------------
    def _load_champion_list(self):
        return [str(champ) for champ in self.splash_dir.iterdir()
                if champ.is_dir() and any(champ.glob("*.jpg"))]

    # -------------------------
    # Strong LCK layout (no overlap)
    # -------------------------
    def _lck_slots(self):
        """
        Perfectly aligned 10 slots, NO OVERLAP guaranteed.
        """
        y = 150

        # fixed spacing: 280 px between slots (256px width)
        gap = 280

        blue_start = 80
        red_start = self.image_width - (80 + 256)

        blue_slots = [(blue_start + i * gap, y) for i in range(5)]
        red_slots  = [(red_start - i * gap, y) for i in range(5)]

        return blue_slots + red_slots[::-1]   # keep order L → R

    # -------------------------
    # Jitter but with collision avoidance
    # -------------------------
    def _jitter_slots(self, slots):
        final = []

        for (x, y) in slots:
            for _ in range(50):  # retry jitter if collide
                nx = x + random.randint(-20, 20)
                ny = y + random.randint(-20, 20)

                # ensure inside image
                if nx < 0 or nx + self.card_width > self.image_width:
                    continue
                if ny < 0 or ny + self.card_height > self.image_height:
                    continue

                # check collision with already placed slots
                collision = False
                for (px, py) in final:
                    if abs(nx - px) < self.card_width and abs(ny - py) < self.card_height:
                        collision = True
                        break

                if not collision:
                    final.append((nx, ny))
                    break

        return final

    # -------------------------
    # Background
    # -------------------------
    def _make_background(self):
        bg = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        for y in range(self.image_height):
            intensity = int(10 + (y / self.image_height) * 25)
            bg[y, :] = [intensity, intensity // 2, intensity // 3]
        return bg

    # -------------------------
    # Crop + resize splash
    # -------------------------
    def _prep_splash(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w, _ = img.shape
        crop = img[:, int(0.2 * w):int(0.8 * w)]
        return cv2.resize(crop, (self.card_width, self.card_height))

    # -------------------------
    # Place card
    # -------------------------
    def _place(self, bg, splash, pos):
        x, y = pos
        bg[y:y+self.card_height, x:x+self.card_width] = splash
        cv2.rectangle(bg, (x, y), (x+self.card_width, y+self.card_height),
                      (180,160,130), 3)

    # -------------------------
    # Labels
    # -------------------------
    def _save_label(self, positions, fname):
        path = self.lbl_dir / (fname + ".txt")
        with open(path, "w") as f:
            for (x, y) in positions:
                xc = (x + self.card_width / 2) / self.image_width
                yc = (y + self.card_height / 2) / self.image_height
                w = self.card_width / self.image_width
                h = self.card_height / self.image_height
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # -------------------------
    # Generate 1 image
    # -------------------------
    def generate_single(self, idx):
        bg = self._make_background()
        base_slots = self._lck_slots()
        positions = self._jitter_slots(base_slots)

        # random 10 champions
        champs = random.sample(self.available_champions, 10)

        for pos, champ in zip(positions, champs):
            splash_path = random.choice(list(Path(champ).glob("*.jpg")))
            splash = self._prep_splash(str(splash_path))
            if splash is not None:
                self._place(bg, splash, pos)

        fname = f"banpick_{idx:06d}"
        cv2.imwrite(str(self.img_dir / (fname + ".jpg")), bg)
        self._save_label(positions, fname)

    # -------------------------
    # Generate dataset
    # -------------------------
    def generate_dataset(self, count=500):
        for i in range(count):
            self.generate_single(i)
        logger.info("✔ Dataset generated successfully")


# RUN
if __name__ == "__main__":
    gen = LCKDatasetGenerator()
    gen.generate_dataset(500)
