"""
Скрипт для финальной многоступенчатой нормализации изображений.

Пайплайн нормализации:
1. Median Blur - удаление мелкого шума
2. Gamma Correction - осветление темных областей
3. CLAHE - усиление локального контраста
4. Z-Score Normalization - стандартизация интенсивности пикселей

Структура:
- Входные данные: data/03_augmented
- Выходные данные: data/04_normalized
- Метки копируются без изменений
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

# Параметры конвейера нормализации
MEDIAN_KSIZE = 3                    # Размер ядра медианного фильтра
GAMMA = 0.5                         # Коэффициент гамма-коррекции
CLAHE_CLIP_LIMIT = 3.0              # Предел контраста для CLAHE
CLAHE_GRID_SIZE = (30, 30)          # Размер сетки для CLAHE


# ============================================================================
# ФУНКЦИИ НОРМАЛИЗАЦИИ
# ============================================================================

def apply_normalization_pipeline(
    image: np.ndarray,
    clahe_processor
) -> np.ndarray:
    """
    Применяет полный конвейер обработки к одному изображению.
    
    Args:
        image: Входное изображение (grayscale)
        clahe_processor: Объект cv2.CLAHE
    
    Returns:
        Нормализованное изображение
    """
    processed_image = image.copy()

    # ШАГ 1: Медианное размытие для удаления шума
    processed_image = cv2.medianBlur(processed_image, MEDIAN_KSIZE)

    # ШАГ 2: Гамма-коррекция для осветления темных участков
    inv_gamma = 1.0 / GAMMA
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    processed_image = cv2.LUT(processed_image, table)

    # ШАГ 3: Применение CLAHE для улучшения локального контраста
    processed_image = clahe_processor.apply(processed_image)
    
    # ШАГ 4: Z-Score нормализация для стандартизации интенсивности
    mean, std = cv2.meanStdDev(processed_image)
    
    # Защита от деления на ноль для пустых изображений
    if std[0, 0] > 1e-6:
        processed_image = (processed_image - mean[0, 0]) / std[0, 0]
    
    # Преобразование обратно в 8-битный формат [0, 255]
    processed_image = cv2.normalize(
        processed_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    return processed_image


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция для запуска процесса нормализации."""
    
    # Проверка существования входной директории
    if not INPUT_DATA_DIR.exists():
        print(
            f"Error: Input directory {INPUT_DATA_DIR} not found. "
            f"Please run augment_data.py first."
        )
        return

    # Подготовка выходной директории
    if OUTPUT_DATA_DIR.exists():
        print(
            f"Removing existing normalized data directory: "
            f"{OUTPUT_DATA_DIR}"
        )
        shutil.rmtree(OUTPUT_DATA_DIR)
    
    print(f"Creating new normalized data directory: {OUTPUT_DATA_DIR}")

    # Создание объекта CLAHE один раз для эффективности
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    # Обработка каждого split
    for split in ["train", "valid", "test"]:
        input_split_dir = INPUT_DATA_DIR / split
        
        if not input_split_dir.exists():
            print(
                f"Split '{split}' not found in {INPUT_DATA_DIR}, skipping."
            )
            continue
        
        print(f"\n{'='*60}")
        print(f"Normalizing '{split}' split")
        print(f"{'='*60}")
        
        # Настройка путей
        input_img_dir = input_split_dir / "images"
        input_label_dir = input_split_dir / "labels"
        
        output_img_dir = OUTPUT_DATA_DIR / split / "images"
        output_label_dir = OUTPUT_DATA_DIR / split / "labels"
        
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # Получение списка изображений
        image_paths = sorted(
            list(input_img_dir.glob("*.jpg")) +
            list(input_img_dir.glob("*.jpeg")) +
            list(input_img_dir.glob("*.png"))
        )

        # Обработка каждого изображения
        for img_path in tqdm(
            image_paths,
            desc=f"Normalizing {split} images"
        ):
            # Чтение изображения в grayscale
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(
                    f"Warning: Could not read image {img_path}. Skipping."
                )
                continue
            
            # Применение полного конвейера нормализации
            normalized_image = apply_normalization_pipeline(image, clahe)
            
            # Сохранение нормализованного изображения в формате PNG
            output_path = output_img_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(output_path), normalized_image)
            
            # Копирование соответствующего файла метки
            label_path = input_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output_label_dir)

    print("\n" + "="*60)
    print("Normalization script finished successfully!")
    print("="*60)


if __name__ == "__main__":
    main()