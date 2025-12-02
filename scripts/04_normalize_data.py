"""
Универсальный скрипт для предобработки (нормализации) изображений.

Поддерживает два режима работы:
1.  `dataset`: Обрабатывает структурированный набор данных в формате YOLO
    (например, `train/images`, `train/labels`). Метки (`.txt`) копируются
    без изменений.
2.  `flat`: Рекурсивно обрабатывает все изображения в указанной директории,
    сохраняя структуру подпапок. Метки игнорируются.

Пример использования:
- Для набора данных YOLO:
  `python 04_normalize_data.py --mode dataset --input data/01_raw --output data/02_normalized`
- Для папки с изображениями:
  `python 04_normalize_data.py --mode flat --input /path/to/images --output /path/to/output`
"""

import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm import tqdm
from typing import List

# ============================================================================ 
# ⚙️ КОНФИГУРАЦИЯ ОБРАБОТКИ
# ============================================================================ 

@dataclass
class ProcessingConfig:
    """Параметры для пайплайна нормализации изображений."""
    # Bilateral Filter (удаление шума с сохранением краев)
    use_bilateral: bool = True
    bilateral_d: int = 5
    bilateral_sigma_color: int = 100
    bilateral_sigma_space: int = 80

    # Median Blur (альтернативное удаление шума)
    use_median: bool = False
    median_ksize: int = 3

    # CLAHE (локальное выравнивание гистограммы для повышения контраста)
    clahe_clip_limit: float = 3.0
    clahe_grid_size: tuple[int, int] = (32, 32)

    # Gamma Correction (коррекция яркости)
    use_gamma: bool = True
    gamma_value: float = 1.60

    # Sharpening (повышение резкости)
    use_sharpen: bool = True
    sharpen_alpha: float = 0.40

# ============================================================================ 
# 🛠 ПАЙПЛАЙН ОБРАБОТКИ
# ============================================================================ 

def apply_normalization_pipeline(
    image: np.ndarray, config: ProcessingConfig, clahe_processor: cv2.CLAHE
) -> np.ndarray:
    """
    Применяет к изображению последовательность фильтров для нормализации.
    """
    processed = image.copy()

    if config.use_bilateral:
        processed = cv2.bilateralFilter(
            processed, config.bilateral_d, config.bilateral_sigma_color, config.bilateral_sigma_space
        )
    if config.use_median:
        processed = cv2.medianBlur(processed, config.median_ksize)

    processed = clahe_processor.apply(processed)

    if config.use_gamma:
        table = np.array(
            [((i / 255.0) ** config.gamma_value) * 255 for i in np.arange(256)]
        ).astype("uint8")
        processed = cv2.LUT(processed, table)

    if config.use_sharpen:
        gaussian = cv2.GaussianBlur(processed, (0, 0), 3.0)
        processed = cv2.addWeighted(
            processed, 1.0 + config.sharpen_alpha, gaussian, -config.sharpen_alpha, 0
        )

    # Z-Score нормализация и масштабирование до 0-255
    processed_float = processed.astype(np.float32)
    mean, std = cv2.meanStdDev(processed_float)
    if std[0, 0] > 1e-6:
        processed_float = (processed_float - mean[0, 0]) / std[0, 0]

    return cv2.normalize(
        processed_float, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

# ============================================================================ 
# 📂 ЛОГИКА РАБОТЫ С ФАЙЛАМИ
# ============================================================================ 

def process_single_file(
    img_path: Path, output_dir: Path, config: ProcessingConfig, clahe: cv2.CLAHE
):
    """Читает, обрабатывает и сохраняет одно изображение."""
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  [Предупреждение] Не удалось прочитать: {img_path.name}")
        return

    normalized_image = apply_normalization_pipeline(image, config, clahe)
    output_path = output_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(output_path), normalized_image)

def process_dataset_mode(
    input_dir: Path, output_dir: Path, config: ProcessingConfig, clahe: cv2.CLAHE
):
    """Обрабатывает набор данных в формате YOLO (train/valid/test)."""
    print(f"🔹 Режим: Dataset. Обработка {input_dir.name}...")
    for split in ["train", "valid", "test"]:
        input_img_dir = input_dir / "images" / split
        input_label_dir = input_dir / "labels" / split

        if not input_img_dir.is_dir(): # Проверяем существование директории с изображениями
            continue

        print(f"  📂 Обработка набора '{split}'...")
        output_img_dir = output_dir / "images" / split
        output_label_dir = output_dir / "labels" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        if input_label_dir.is_dir():
            shutil.copytree(input_label_dir, output_label_dir, dirs_exist_ok=True)

        image_paths = sorted(list(input_img_dir.glob("*.jpg")) + list(input_img_dir.glob("*.png")))
        for img_path in tqdm(image_paths, desc=f"  -> {split}"):
            process_single_file(img_path, output_img_dir, config, clahe)

def process_flat_mode(
    input_dir: Path, output_dir: Path, config: ProcessingConfig, clahe: cv2.CLAHE
):
    """Рекурсивно обрабатывает все изображения в директории."""
    print(f"🔹 Режим: Flat. Рекурсивная обработка {input_dir}...")
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    image_paths = [p for ext in extensions for p in input_dir.rglob(ext)]

    if not image_paths:
        print(f"⚠️ Изображения не найдены в {input_dir}")
        return

    for img_path in tqdm(image_paths, desc="  -> Изображения"):
        relative_path = img_path.relative_to(input_dir)
        save_dir = output_dir / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        process_single_file(img_path, save_dir, config, clahe)

# ============================================================================ 
# 🚀 ЗАПУСК
# ============================================================================ 

def main():
    """Главная функция для парсинга аргументов и запуска обработки."""
    parser = argparse.ArgumentParser(description="Скрипт нормализации изображений.")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["dataset", "flat"],
        help="Режим работы: 'dataset' для YOLO-структуры, 'flat' для папки с картинками."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Путь к входной директории."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Путь к выходной директории."
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Ошибка: Входная директория не найдена: {args.input}")
        return

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    config = ProcessingConfig()
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid_size
    )

    print(f"🚀 Старт обработки: {args.input} -> {args.output}")
    print(f"⚙️ Конфигурация: CLAHE={config.clahe_clip_limit}, Gamma={config.gamma_value}, Sharpen={config.sharpen_alpha}")

    if args.mode == "dataset":
        process_dataset_mode(args.input, args.output, config, clahe)
    elif args.mode == "flat":
        process_flat_mode(args.input, args.output, config, clahe)

    print("\n✅ Обработка завершена.")

if __name__ == "__main__":
    main()