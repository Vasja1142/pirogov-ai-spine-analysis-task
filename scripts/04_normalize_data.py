"""
Скрипт предобработки данных v2.0 (на основе параметров Tuner).
Включает: Bilateral Filter, CLAHE, Gamma, Sharpening, Z-Score Normalization.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ============================================================================
# ⚙️ КОНФИГУРАЦИЯ (ВСЕ НАСТРОЙКИ СО СКРИНШОТА)
# ============================================================================

INPUT_DATA_DIR = Path("data/03_augmented")
OUTPUT_DATA_DIR = Path("data/04_normalized")

# 1. Шум и Детали
USE_BILATERAL = True                # Включено на скрине
BILATERAL_D = 5                     # Diameter
BILATERAL_SIGMA_COLOR = 100         # Sigma Color
BILATERAL_SIGMA_SPACE = 80          # Sigma Space

USE_MEDIAN = False                  # Снята галочка на скрине
MEDIAN_KSIZE = 3

# 2. Контраст и Яркость
CLAHE_CLIP_LIMIT = 3.0             # Со скрина
CLAHE_GRID_SIZE = (32, 32)          # Со скрина

USE_GAMMA = True
GAMMA_VALUE = 2.00                  # 206 со слайдера = 2.06 (темнее/контрастнее)

# 3. Резкость (Sharpen)
USE_SHARPEN = True
SHARPEN_ALPHA = 0.60                # Со скрина

# ============================================================================
# 🛠 ПАЙПЛАЙН ОБРАБОТКИ
# ============================================================================

def apply_normalization_pipeline(image: np.ndarray, clahe_processor) -> np.ndarray:
    processed_image = image.copy()

    # 1. Bilateral Filter (Сглаживание с сохранением краев)
    if USE_BILATERAL:
        processed_image = cv2.bilateralFilter(
            processed_image, 
            d=BILATERAL_D, 
            sigmaColor=BILATERAL_SIGMA_COLOR, 
            sigmaSpace=BILATERAL_SIGMA_SPACE
        )

    # 2. Median Blur (если нужен)
    if USE_MEDIAN:
        processed_image = cv2.medianBlur(processed_image, MEDIAN_KSIZE)

    # 3. CLAHE (Локальный контраст)
    processed_image = clahe_processor.apply(processed_image)

    # 4. Gamma Correction
    # Формула: O = (I / 255) ^ gamma * 255
    if USE_GAMMA:
        # Создаем таблицу поиска (LUT) для скорости
        inv_gamma = GAMMA_VALUE # Albumentations использует значение напрямую как степень
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        processed_image = cv2.LUT(processed_image, table)

    # 5. Sharpening (Повышение резкости)
    # Метод Unsharp Mask: Original + (Original - Blurred) * Amount
    if USE_SHARPEN:
        gaussian = cv2.GaussianBlur(processed_image, (0, 0), 3.0)
        processed_image = cv2.addWeighted(processed_image, 1.0 + SHARPEN_ALPHA, gaussian, -SHARPEN_ALPHA, 0)

    # 6. Z-Score Нормализация (Стандартизация)
    # Приводим к нулевому среднему и единичному отклонению, затем обратно в 0-255
    # Это помогает нейросети лучше сходиться.
    processed_image = processed_image.astype("float32")
    mean, std = cv2.meanStdDev(processed_image)
    
    if std[0, 0] > 1e-6:
        processed_image = (processed_image - mean[0, 0]) / std[0, 0]
    
    # Масштабируем обратно в 0-255 для сохранения в файл
    processed_image = cv2.normalize(
        processed_image, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")

    return processed_image


# ============================================================================
# 🚀 ЗАПУСК
# ============================================================================

def main():
    if not INPUT_DATA_DIR.exists():
        print(f"❌ Ошибка: Папка {INPUT_DATA_DIR} не найдена.")
        return

    if OUTPUT_DATA_DIR.exists():
        shutil.rmtree(OUTPUT_DATA_DIR)
    
    print(f"🚀 Начинаю обработку данных в: {OUTPUT_DATA_DIR}")
    print(f"⚙️ Параметры: Bilateral={USE_BILATERAL}, CLAHE={CLAHE_CLIP_LIMIT}, Gamma={GAMMA_VALUE}, Sharpen={SHARPEN_ALPHA}")

    # Инициализация CLAHE один раз
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    for split in ["train", "valid", "test"]:
        input_split_dir = INPUT_DATA_DIR / split
        if not input_split_dir.exists(): continue
        
        print(f"📂 Обработка папки '{split}'...")
        
        input_img_dir = input_split_dir / "images"
        input_label_dir = input_split_dir / "labels"
        output_img_dir = OUTPUT_DATA_DIR / split / "images"
        output_label_dir = OUTPUT_DATA_DIR / split / "labels"
        
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(list(input_img_dir.glob("*.jpg")) + list(input_img_dir.glob("*.png")))

        for img_path in tqdm(image_paths):
            # Читаем картинку (сразу в оттенках серого)
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            
            # Применяем весь пайплайн
            normalized_image = apply_normalization_pipeline(image, clahe)
            
            # Сохраняем всегда в PNG (чтобы не терять качество на сжатии JPG)
            output_path = output_img_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(output_path), normalized_image)
            
            # Просто копируем метки (они не меняются от изменения цвета/яркости)
            label_path = input_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output_label_dir)

    print("\n✅ Обработка завершена успешно.")


if __name__ == "__main__":
    main()