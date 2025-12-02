"""
Скрипт для предварительной обработки набора данных изображений.

Этот скрипт выполняет следующие шаги:
1.  Читает необработанные изображения и соответствующие им файлы меток.
2.  Обрезает изображения по вертикали на основе расположения многоугольников разметки,
    добавляя отступы.
3.  Пересчитывает координаты меток в соответствии с новыми размерами обрезанного изображения.
4.  Изменяет размер обработанных изображений до целевого размера с сохранением
    соотношения сторон.
5.  Разделяет набор данных на обучающую и тестовую выборки.
6.  Сохраняет обработанные изображения и метки в структурированную выходную директорию.
7.  Создает файл `dataset.yaml` для использования с фреймворками машинного обучения,
    такими как YOLO.
"""

import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Set

# ================= НАСТРОЙКИ =================
# Входная директория с исходными изображениями и .txt файлами разметки
INPUT_DIR = Path("data/01_raw/no_label/for_labeling/images")
# Выходная директория для обработанных данных
OUTPUT_DIR = Path("data/02_processed")
# Отступ от крайних точек разметки при обрезке (в пикселях)
PADDING: int = 50
# Целевой размер для меньшей стороны изображения после масштабирования
TARGET_MIN_DIM: int = 640
# Доля данных, которая будет использоваться для тестового набора (например, 0.2 для 20%)
TEST_RATIO: float = 0.2
# Сид для воспроизводимости случайного разделения на train/test
SEED: int = 42
# Расширения файлов изображений, которые будут обрабатываться
EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp'}
# =============================================


def create_structure(base_dir: Path) -> None:
    """Создает необходимую структуру директорий для набора данных."""
    for split in ["train", "test"]:
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def get_files_map(input_path: Path) -> Dict[str, Tuple[Path, Path]]:
    """Находит пары изображений и файлов меток."""
    files_map = {}
    for f in input_path.iterdir():
        if f.suffix.lower() in EXTENSIONS:
            txt_file = f.with_suffix(".txt")
            if txt_file.exists():
                files_map[f.name] = (f, txt_file)
    return files_map

def process_single_image(
    img_path: Path, txt_path: Path
) -> Tuple[np.ndarray, List[str]]:
    """Обрабатывает одно изображение: обрезка, пересчет меток и изменение размера."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    h_orig, w_orig, _ = img.shape

    with open(txt_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None, None

    polygons = []
    min_y_abs, max_y_abs = h_orig, 0

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id = int(parts[0])
        coords = parts[1:]
        ys_abs = [y * h_orig for y in coords[1::2]]
        min_y_abs = min(min_y_abs, min(ys_abs))
        max_y_abs = max(max_y_abs, max(ys_abs))
        polygons.append({"cls": cls_id, "points": coords})

    crop_y1 = int(max(0, min_y_abs - PADDING))
    crop_y2 = int(min(h_orig, max_y_abs + PADDING))

    if crop_y2 <= crop_y1:
        return None, None

    cropped_img = img[crop_y1:crop_y2, :]
    h_crop, w_crop, _ = cropped_img.shape

    new_labels_lines = []
    for poly in polygons:
        new_coords = []
        old_coords = poly["points"]
        for i in range(0, len(old_coords), 2):
            x = old_coords[i]
            y = old_coords[i + 1]
            abs_y_orig = y * h_orig
            abs_y_crop = abs_y_orig - crop_y1
            new_y = max(0.0, min(1.0, abs_y_crop / h_crop))
            new_coords.extend([f"{x:.6f}", f"{new_y:.6f}"])
        new_labels_lines.append(f"{poly['cls']} {" ".join(new_coords)}")

    scale_factor = TARGET_MIN_DIM / min(h_crop, w_crop)
    new_w = int(w_crop * scale_factor)
    new_h = int(h_crop * scale_factor)
    resized_img = cv2.resize(
        cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    return resized_img, new_labels_lines

def save_data(
    split_name: str, items: List[Dict], output_dir: Path
) -> None:
    """Сохраняет обработанные изображения и метки на диск."""
    print(f"Сохранение {split_name}: {len(items)} файлов...")
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    for item in items:
        base_name = item["name"]
        cv2.imwrite(str(img_dir / f"{base_name}.jpg"), item["img"])
        with open(lbl_dir / f"{base_name}.txt", "w") as f:
            f.write("\n".join(item["labels"]))

def create_yaml_config(output_dir: Path) -> None:
    """Создает файл конфигурации dataset.yaml."""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(f"# Путь к данным относительно корня проекта\n")
        f.write(f"path: ../{output_dir.as_posix()}\n\n")
        f.write("# Пути к изображениям для обучения и валидации\n")
        f.write("train: images/train\n")
        f.write("val: images/test\n\n")
        f.write("# Имена классов\n")
        f.write("names:\n")
        f.write("  0: object\n")

def main():
    """Основная функция для запуска всего процесса обработки."""
    random.seed(SEED)

    files_map = get_files_map(INPUT_DIR)
    if not files_map:
        print(f"Файлы не найдены в {INPUT_DIR}. Проверьте путь.")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    create_structure(OUTPUT_DIR)

    valid_data = []
    print(f"Найдено {len(files_map)} пар изображений/меток. Начинаем обработку...")

    for img_name, (img_path, txt_path) in tqdm(files_map.items()):
        resized_img, new_labels = process_single_image(img_path, txt_path)
        if resized_img is not None and new_labels:
            valid_data.append(
                {"name": img_path.stem, "img": resized_img, "labels": new_labels}
            )

    print(f"Успешно обработано: {len(valid_data)} изображений.")

    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * (1 - TEST_RATIO))
    splits = {"train": valid_data[:split_idx], "test": valid_data[split_idx:]}

    for split_name, items in splits.items():
        save_data(split_name, items, OUTPUT_DIR)

    create_yaml_config(OUTPUT_DIR)
    print("Готово!")


if __name__ == "__main__":
    main()