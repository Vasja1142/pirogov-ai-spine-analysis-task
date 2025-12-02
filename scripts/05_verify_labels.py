"""
Скрипт для визуальной проверки корректности разметки набора данных.

Функциональность:
- Выбирает случайные изображения из указанного набора данных (например, 'train').
- Загружает соответствующие файлы разметки в формате YOLO (полигоны или bounding box).
- Отрисовывает разметку поверх изображений.
- Сохраняет результирующие изображения в отдельную директорию для проверки.
- Поддерживает загрузку имен классов из `dataset.yaml`.
"""

import argparse
import random
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np
from typing import List, Dict, Union, Optional

# ============================================================================
# ⚙️ КОНФИГУРАЦИЯ И СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class LabelData:
    """Структура для хранения информации о разметке одного объекта."""
    class_id: int
    # Координаты могут быть полигоном (Nx2) или bounding box (1x4)
    coords: np.ndarray
    class_name: str = ""

@dataclass
class ImageData:
    """Структура для хранения полного набора данных одного изображения."""
    image_path: Path
    labels: List[LabelData] = field(default_factory=list)

# ============================================================================
# 🎨 ФУНКЦИИ ОТРИСОВКИ
# ============================================================================

def draw_polygon(image: np.ndarray, points: np.ndarray, text: str, color: tuple):
    """Отрисовывает на изображении один полигон и его метку."""
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    if points.size > 0:
        # Размещаем текст рядом с первой точкой полигона
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_bounding_box(image: np.ndarray, box: np.ndarray, text: str, color: tuple):
    """Отрисовывает на изображении один bounding box и его метку."""
    x1, y1, x2, y2 = box.astype(int).flatten()
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # Размещаем текст над верхним левым углом бокса
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def visualize_labels(
    image_data: ImageData, class_names: Dict[int, str], output_dir: Path
):
    """
    Загружает изображение, отрисовывает все его метки и сохраняет результат.
    """
    image = cv2.imread(str(image_data.image_path))
    if image is None:
        print(f"  [Предупреждение] Не удалось прочитать: {image_data.image_path.name}")
        return

    h, w = image.shape[:2]

    if not image_data.labels:
        cv2.putText(image, "NO LABELS FOUND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for label in image_data.labels:
            class_name = class_names.get(label.class_id, f"ID:{label.class_id}")
            denormalized_coords = label.coords.copy()
            denormalized_coords[:, 0] *= w
            denormalized_coords[:, 1] *= h

            if len(label.coords.flatten()) == 4: # Bounding box
                # Конвертация из (center_x, center_y, width, height) в (x1, y1, x2, y2)
                cx, cy, bw, bh = denormalized_coords.flatten()
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                box = np.array([[x1, y1, x2, y2]])
                draw_bounding_box(image, box, class_name, (0, 255, 0))
            else: # Polygon
                points = denormalized_coords.astype(np.int32)
                draw_polygon(image, points, class_name, (0, 255, 255))

    save_path = output_dir / f"verify_{image_data.image_path.name}"
    cv2.imwrite(str(save_path), image)
    print(f"  -> Сохранено: {save_path.name}")

# ============================================================================
# 📂 ЛОГИКА РАБОТЫ С ФАЙЛАМИ
# ============================================================================

def load_class_names(yaml_path: Path) -> Dict[int, str]:
    """Загружает имена классов из YAML файла."""
    if not yaml_path.is_file():
        print(f"⚠️  YAML файл не найден: {yaml_path}. Будут использоваться только ID классов.")
        return {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            names = data.get('names', {})
            if isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
            if isinstance(names, dict):
                return names
            print("⚠️  Некорректный формат 'names' в YAML, ожидался список или словарь.")
            return {}
    except Exception as e:
        print(f"❌ Ошибка чтения YAML файла: {e}")
        return {}

def load_image_data(
    image_dir: Path, label_dir: Path
) -> List[ImageData]:
    """Загружает пути к изображениям и соответствующие им данные разметки."""
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    all_data = []

    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        image_data = ImageData(image_path=img_path)

        if label_path.is_file():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    class_id = int(parts[0])
                    coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                    image_data.labels.append(LabelData(class_id=class_id, coords=coords))
        all_data.append(image_data)
    return all_data

# ============================================================================
# 🚀 ЗАПУСК
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Скрипт для визуальной проверки разметки.")
    parser.add_argument(
        "--base-dir", type=Path, default=Path("data/03_augmented"),
        help="Основная директория с набором данных."
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Набор для проверки (train, valid, test)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/06_verification_runs"),
        help="Директория для сохранения проверочных изображений."
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Количество случайных изображений для проверки."
    )
    args = parser.parse_args()

    image_dir = args.base_dir / "images" / args.split
    label_dir = args.base_dir / "labels" / args.split
    yaml_path = args.base_dir / "dataset.yaml"

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"❌ Ошибка: Директории images/{args.split} или labels/{args.split} не найдены в {args.base_dir}")
        return

    # Подготовка выходной директории
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Результаты будут сохранены в: {args.output_dir.resolve()}")

    # Загрузка данных
    class_names = load_class_names(yaml_path)
    all_image_data = load_image_data(image_dir, label_dir)

    if not all_image_data:
        print(f"⚠️ В директории {image_dir} не найдено изображений для проверки.")
        return

    # Выборка и визуализация
    num_samples = min(len(all_image_data), args.samples)
    if num_samples == 0:
        print("Нет данных для выборки.")
        return
        
    print(f"✅ Найдено {len(all_image_data)} изображений. Выбираем {num_samples} для проверки...")
    selected_samples = random.sample(all_image_data, num_samples)

    for sample in selected_samples:
        visualize_labels(sample, class_names, args.output_dir)

    print(f"\n🎉 Проверка завершена. Просмотрите изображения в папке '{args.output_dir}'.")

if __name__ == "__main__":
    main()