"""
Упрощенный скрипт для нормализации изображений.

Этот скрипт применяет последовательность из трех фильтров для улучшения
качества изображений:
1.  **Bilateral Filter**: Умное сглаживание шума с сохранением контуров.
2.  **Robust Auto-Levels**: Растягивание гистограммы с отсечением выбросов для
    повышения контраста.
3.  **Unsharp Mask**: Увеличение контурной резкости.

Скрипт копирует структуру входной директории в выходную, обрабатывая все
найденные изображения и копируя файлы меток (.txt) без изменений.
"""

import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm import tqdm
from typing import Set

# ============================================================================
# ⚙️ КОНФИГУРАЦИЯ ОБРАБОТКИ
# ============================================================================

@dataclass
class SimpleProcessingConfig:
    """Параметры для упрощенного пайплайна нормализации."""
    # 1. Bilateral Filter
    bilat_d: int = 3
    bilat_sigma_color: int = 75
    bilat_sigma_space: int = 75

    # 2. Robust Auto-Levels
    cutoff_percent: float = 0.03

    # 3. Unsharp Mask
    sharpen_sigma: float = 5.0
    sharpen_amount: float = 0.5

# ============================================================================
# 🛠 ПАЙПЛАЙН ОБРАБОТКИ
# ============================================================================

def apply_simple_pipeline(
    image: np.ndarray, config: SimpleProcessingConfig
) -> np.ndarray:
    """
    Применяет к изображению упрощенную последовательность фильтров.
    Работает как с цветными, так и с Ч/Б изображениями.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # Для цветных изображений обрабатываем только канал яркости (L)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        processed_l = process_channel(l_channel, config)
        merged_lab = cv2.merge([processed_l, a_channel, b_channel])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    else:
        # Для Ч/Б изображений обрабатываем напрямую
        return process_channel(image, config)


def process_channel(
    channel: np.ndarray, config: SimpleProcessingConfig
) -> np.ndarray:
    """
    Применяет полный набор фильтров к одному каналу изображения.

    Args:
        channel: Одноканальное 8-битное изображение (например, grayscale или L-канал).
        config: Конфигурация с параметрами фильтров.

    Returns:
        Обработанный канал.
    """
    # 1. Сглаживание шума
    bilateral_filtered = cv2.bilateralFilter(
        channel, config.bilat_d, config.bilat_sigma_color, config.bilat_sigma_space
    )

    # 2. Растягивание гистограммы
    auto_leveled = robust_auto_levels(bilateral_filtered, config.cutoff_percent)

    # 3. Повышение резкости
    sharpened = unsharp_mask(auto_leveled, config.sharpen_sigma, config.sharpen_amount)

    return sharpened


def robust_auto_levels(channel: np.ndarray, cutoff: float) -> np.ndarray:
    """Растягивает гистограмму канала, отсекая выбросы."""
    channel_float = channel.astype(np.float32)
    low_val = np.percentile(channel_float, cutoff)
    high_val = np.percentile(channel_float, 100 - cutoff)

    if high_val <= low_val:
        return channel # Избегаем деления на ноль

    clipped = np.clip(channel_float, low_val, high_val)
    normalized = (clipped - low_val) / (high_val - low_val) * 255.0
    return normalized.astype(np.uint8)


def unsharp_mask(channel: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Применяет фильтр нерезкого маскирования для повышения резкости."""
    gaussian = cv2.GaussianBlur(channel, (0, 0), sigma)
    return cv2.addWeighted(channel, 1.0 + amount, gaussian, -amount, 0)

# ============================================================================
# 🚀 ЗАПУСК
# ============================================================================

def main():
    """Главная функция для парсинга аргументов и запуска обработки."""
    parser = argparse.ArgumentParser(
        description="Упрощенный скрипт нормализации изображений (Bilateral -> Levels -> Sharpen)."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/03_augmented"),
        help="Путь к входной директории с данными."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/04_normalized"),
        help="Путь к выходной директории для сохранения результатов."
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"❌ Ошибка: Входная директория не найдена: {args.input}")
        return

    # Подготовка выходной директории
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    # Копирование только файлов меток и создание структуры папок
    txt_files = list(args.input.rglob("*.txt"))
    if txt_files:
        print(f"Копирование {len(txt_files)} файлов меток...")
        for txt_file in txt_files:
            relative_path = txt_file.relative_to(args.input)
            output_txt_path = args.output / relative_path
            output_txt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt_file, output_txt_path)

    # Поиск изображений для обработки
    image_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = [p for ext in image_extensions for p in args.input.rglob(f"*{ext}")]

    if not all_images:
        print(f"⚠️ Изображения для обработки не найдены в {args.input}")
        return

    config = SimpleProcessingConfig()
    print(f"🚀 Найдено {len(all_images)} изображений. Запуск пайплайна...")

    for img_path in tqdm(all_images, desc="Обработка изображений"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [Предупреждение] Не удалось прочитать: {img_path.name}")
            continue

        processed_img = apply_simple_pipeline(image, config)

        relative_path = img_path.relative_to(args.input)
        output_img_path = args.output / relative_path
        output_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем с тем же именем и расширением
        cv2.imwrite(str(output_img_path), processed_img)

    print(f"\n✅ Обработка завершена. Результаты сохранены в: {args.output}")

if __name__ == "__main__":
    main()