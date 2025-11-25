"""
Скрипт для обработки и объединения наборов данных СЕГМЕНТАЦИИ.

Изменения:
- Источник данных: data/01_raw/segmentation
- Специфичная логика: Если датасет == 'Scoliosis.v2i.yolov12', 
  выполняется обрезка изображения сверху (над самым верхним позвонком).
- Для остальных датасетов изображение остается оригинальным.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

INPUT_DATA_ROOT = Path("data/01_raw/segmentation")
UNIFIED_OUTPUT_ROOT = Path("data/02_processed")

# Насколько выше самого верхнего позвонка резать (в процентах от высоты)
CROP_MARGIN = 0.02 

# Глобальный список классов 
GLOBAL_CLASSES = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5'
]

GLOBAL_CLASS_MAP = {name: i for i, name in enumerate(GLOBAL_CLASSES)}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_image_files(path: Path) -> list[Path]:
    return sorted([
        p for p in path.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

def create_class_remapping(local_yaml_path: Path) -> dict:
    if not local_yaml_path.exists():
        return None
    
    with open(local_yaml_path, 'r') as f:
        local_data = yaml.safe_load(f)
    
    local_classes = local_data.get('names', [])
    remapping = {}
    
    for local_id, name in enumerate(local_classes):
        clean_name = name.strip().upper()
        if clean_name in GLOBAL_CLASS_MAP:
            remapping[local_id] = GLOBAL_CLASS_MAP[clean_name]
        else:
            # Поиск при неточном совпадении
            for g_name in GLOBAL_CLASSES:
                if g_name == clean_name:
                    remapping[local_id] = GLOBAL_CLASS_MAP[g_name]
                    break
    return remapping


def process_and_merge_split(
    source_dataset_dir: Path,
    split_name: str,
    dest_img_dir: Path,
    dest_label_dir: Path,
    class_remapping: dict,
    should_crop: bool
):
    """Обрабатывает один split."""
    dataset_name = source_dataset_dir.name
    
    # Поиск папки (train/val/valid)
    source_split_dir = source_dataset_dir / split_name
    if not source_split_dir.exists():
        if split_name == "valid":
            source_split_dir = source_dataset_dir / "val"
            if not source_split_dir.exists():
                return
        else:
            return

    print(f"--- Processing: {dataset_name}/{split_name} (Crop: {should_crop}) ---")

    source_img_dir = source_split_dir / "images"
    source_label_dir = source_split_dir / "labels"

    image_paths = get_image_files(source_img_dir)

    for img_path in tqdm(image_paths, desc=f"{dataset_name}"):
        label_path = source_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None: continue
        
        orig_h, orig_w = image.shape[:2]
        
        lines = label_path.read_text().strip().split('\n')
        if not lines or not lines[0]: continue

        # 1. Парсим полигоны
        polygons = [] # List of (global_id, [coords...])
        all_y_coords = [] # Для поиска верхней точки

        for line in lines:
            parts = line.split()
            try:
                local_id = int(parts[0])
            except ValueError: continue
            
            if local_id not in class_remapping: continue
            
            global_id = class_remapping[local_id]
            coords = [float(x) for x in parts[1:]]
            
            if len(coords) < 6: continue # Минимум 3 точки
            
            polygons.append((global_id, coords))
            
            # Собираем Y координаты (они на нечетных позициях 1, 3, 5...)
            # coords = [x1, y1, x2, y2, ...]
            ys = coords[1::2]
            all_y_coords.extend(ys)

        if not polygons: continue

        # 2. Вычисляем обрезку
        crop_y_offset = 0
        new_h = orig_h

        if should_crop and all_y_coords:
            # Находим самый верхний пиксель (минимальный Y)
            min_y_norm = min(all_y_coords)
            min_y_abs = min_y_norm * orig_h
            
            # Отступаем вверх на margin
            crop_pos = int(min_y_abs - (orig_h * CROP_MARGIN))
            crop_y_offset = max(0, crop_pos)
            
            # Если отрезать нечего (позвонок и так в самом верху), offset будет 0
            if crop_y_offset > 0:
                image = image[crop_y_offset:, :]
                new_h = image.shape[0]
                if new_h == 0: continue # Защита

        # 3. Пересчитываем координаты меток
        final_labels = []
        
        for global_id, coords in polygons:
            new_coords = []
            # Итерируемся парами (x, y)
            for i in range(0, len(coords), 2):
                x_norm = coords[i]
                y_norm = coords[i+1]
                
                if should_crop and crop_y_offset > 0:
                    # Денормализация Y
                    y_abs = y_norm * orig_h
                    # Сдвиг
                    new_y_abs = y_abs - crop_y_offset
                    # Нормализация к НОВОЙ высоте
                    new_y_norm = new_y_abs / new_h
                    
                    # Клиппинг (чтобы не вылезло за пределы картинки)
                    new_y_norm = max(0.0, min(1.0, new_y_norm))
                    
                    # X не меняется, так как ширину не резали
                    # Но на всякий случай клиппинг
                    x_norm = max(0.0, min(1.0, x_norm))
                    
                    new_coords.extend([x_norm, new_y_norm])
                else:
                    new_coords.extend([x_norm, y_norm])
            
            # Формируем строку
            coords_str = " ".join([f"{c:.6f}" for c in new_coords])
            final_labels.append(f"{global_id} {coords_str}")

        # 4. Сохранение
        new_img_name = f"{dataset_name}_{img_path.name}"
        new_label_name = f"{dataset_name}_{img_path.stem}.txt"
        
        cv2.imwrite(str(dest_img_dir / new_img_name), image)
        (dest_label_dir / new_label_name).write_text('\n'.join(final_labels))


# ============================================================================
# MAIN
# ============================================================================

def main():
    if UNIFIED_OUTPUT_ROOT.exists():
        shutil.rmtree(UNIFIED_OUTPUT_ROOT)
    
    unified_train_images = UNIFIED_OUTPUT_ROOT / "train" / "images"
    unified_train_labels = UNIFIED_OUTPUT_ROOT / "train" / "labels"
    unified_test_images = UNIFIED_OUTPUT_ROOT / "test" / "images"
    unified_test_labels = UNIFIED_OUTPUT_ROOT / "test" / "labels"
    
    for path in [
        unified_train_images, unified_train_labels,
        unified_test_images, unified_test_labels
    ]:
        path.mkdir(parents=True)

    if not INPUT_DATA_ROOT.exists():
        print(f"Error: {INPUT_DATA_ROOT} not found!")
        return

    for dataset_dir in INPUT_DATA_ROOT.iterdir():
        if not dataset_dir.is_dir(): continue
        
        dataset_name = dataset_dir.name
        print(f"\nProcessing dataset: {dataset_name}")

        class_remapping = create_class_remapping(dataset_dir / "data.yaml")
        if not class_remapping: continue

        # --- ЛОГИКА ВКЛЮЧЕНИЯ ОБРЕЗКИ ---
        # Включаем только для конкретного датасета
        should_crop = (dataset_name == "Scoliosis.v2i.yolov12")
        
        process_and_merge_split(
            dataset_dir, "train",
            unified_train_images, unified_train_labels,
            class_remapping, should_crop
        )
        
        process_and_merge_split(
            dataset_dir, "valid",
            unified_test_images, unified_test_labels,
            class_remapping, should_crop
        )

    print("\n" + "="*60)
    print("Done. Cropping applied only to Scoliosis dataset.")
    print("="*60)


if __name__ == "__main__":
    main()