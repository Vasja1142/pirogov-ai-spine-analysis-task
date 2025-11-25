"""
Скрипт для финальной нормализации изображений.
Версия "Basic": Только улучшение качества (Denoise, CLAHE, Normalize).
Инверсия отключена, так как датасет теперь содержит оба варианта (Normal + Inverted).
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

INPUT_DATA_DIR = Path("data/03_augmented")
OUTPUT_DATA_DIR = Path("data/04_normalized")

MEDIAN_KSIZE = 3                    # Размер ядра медианного фильтра
GAMMA = 0.5                         # Коэффициент гамма-коррекции
CLAHE_CLIP_LIMIT = 4.0              # Предел контраста для CLAHE
CLAHE_GRID_SIZE = (30, 30)          # Размер сетки для CLAHE


# ============================================================================
# ПАЙПЛАЙН
# ============================================================================

def apply_normalization_pipeline(image: np.ndarray, clahe_processor) -> np.ndarray:
    processed_image = image.copy()

    # 1. Медианное размытие (Denoise)
    processed_image = cv2.medianBlur(processed_image, MEDIAN_KSIZE)

    # 2. Гамма-коррекция (высветление теней)
    inv_gamma = 1.0 / GAMMA
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    processed_image = cv2.LUT(processed_image, table)

    # 3. CLAHE (Локальный контраст)
    processed_image = clahe_processor.apply(processed_image)
    
    # 4. Z-Score нормализация
    mean, std = cv2.meanStdDev(processed_image)
    if std[0, 0] > 1e-6:
        processed_image = (processed_image - mean[0, 0]) / std[0, 0]
    
    # Возврат в 0-255
    processed_image = cv2.normalize(
        processed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return processed_image


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not INPUT_DATA_DIR.exists():
        print(f"Error: {INPUT_DATA_DIR} not found.")
        return

    if OUTPUT_DATA_DIR.exists():
        shutil.rmtree(OUTPUT_DATA_DIR)
    
    print(f"Normalizing data to: {OUTPUT_DATA_DIR}")

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    for split in ["train", "valid", "test"]:
        input_split_dir = INPUT_DATA_DIR / split
        if not input_split_dir.exists(): continue
        
        print(f"Processing '{split}'...")
        
        input_img_dir = input_split_dir / "images"
        input_label_dir = input_split_dir / "labels"
        output_img_dir = OUTPUT_DATA_DIR / split / "images"
        output_label_dir = OUTPUT_DATA_DIR / split / "labels"
        
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(list(input_img_dir.glob("*.jpg")) + list(input_img_dir.glob("*.png")))

        for img_path in tqdm(image_paths):
            # Читаем сразу в ч/б
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            
            # Обработка
            normalized_image = apply_normalization_pipeline(image, clahe)
            
            # Сохраняем как PNG
            output_path = output_img_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(output_path), normalized_image)
            
            # Копируем метки
            label_path = input_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output_label_dir)

    print("\nNormalization finished.")


if __name__ == "__main__":
    main()