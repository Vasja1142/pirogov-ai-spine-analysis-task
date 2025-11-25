import random
from pathlib import Path
import cv2
import numpy as np
import yaml

# CONFIG
DATA_YAML_PATH = Path("data.yaml")
IMAGES_DIR = Path("data/03_augmented/train/images") # Или 04_normalized
OUTPUT_DIR = Path("data/06_verification_runs")

def main():
    if not DATA_YAML_PATH.exists() or not IMAGES_DIR.exists():
        print("Paths not found.")
        return

    with open(DATA_YAML_PATH, 'r') as f:
        names = yaml.safe_load(f).get('names', [])

    image_paths = list(IMAGES_DIR.glob("*.*"))
    if not image_paths: return
    
    # Берем случайное фото
    img_path = random.choice(image_paths)
    label_path = IMAGES_DIR.parent / "labels" / f"{img_path.stem}.txt"
    
    print(f"Checking: {img_path.name}")
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                # Читаем координаты полигона
                coords = [float(x) for x in parts[1:]]
                points = np.array(coords).reshape(-1, 2)
                
                # Денормализация
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)
                
                # Рисуем полигон
                color = (0, 255, 0) # Зеленый
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
                
                # Текст
                cv2.putText(img, names[cls_id], tuple(points[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / f"verify_{img_path.name}"), img)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()