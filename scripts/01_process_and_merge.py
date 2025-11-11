"""
Скрипт для обработки, унификации классов и объединения наборов данных.

Этапы обработки:
1. Определяет глобальный канонический список классов для всего проекта
2. Создает структуру каталогов в `data/02_processed`
3. Для каждого набора данных в `data/01_raw`:
   - Находит локальный `data.yaml` и читает список классов
   - Создает карту преобразования из локальных ID в глобальные
   - Обрабатывает файлы меток, заменяя локальные ID на глобальные
   - Выполняет обрезку изображений и корректировку координат
4. Сохраняет обработанные файлы в `data/02_processed` с префиксами
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

INPUT_DATA_ROOT = Path("data/01_raw")
UNIFIED_OUTPUT_ROOT = Path("data/02_processed2")
CROP_MARGIN = 0.01  # Дополнительный отступ при обрезке изображений

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
    """Получает список всех изображений в директории."""
    return sorted([
        p for p in path.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def convert_polygon_to_bbox(
    polygon: np.ndarray,
    img_width: int,
    img_height: int
) -> tuple:
    """Конвертирует полигональную аннотацию в bounding box формата YOLO."""
    polygon[:, 0] *= img_width
    polygon[:, 1] *= img_height
    
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    center_x = (x_min + box_width / 2) / img_width
    center_y = (y_min + box_height / 2) / img_height
    
    return center_x, center_y, box_width / img_width, box_height / img_height


def create_class_remapping(local_yaml_path: Path) -> dict:
    """Создает словарь для преобразования локальных ID классов в глобальные."""
    if not local_yaml_path.exists():
        print(f"Warning: {local_yaml_path} not found. Cannot remap classes.")
        return None
    
    with open(local_yaml_path, 'r') as f:
        local_data = yaml.safe_load(f)
    
    local_classes = local_data.get('names', [])
    remapping = {}
    
    for local_id, name in enumerate(local_classes):
        name_upper = name.upper()
        
        if name_upper in GLOBAL_CLASS_MAP:
            remapping[local_id] = GLOBAL_CLASS_MAP[name_upper]
        else:
            print(
                f"Warning: Class '{name}' from {local_yaml_path.parent.name} "
                f"not in GLOBAL_CLASSES. It will be ignored."
            )
    
    return remapping


def process_and_merge_split(
    source_dataset_dir: Path,
    split_name: str,
    dest_img_dir: Path,
    dest_label_dir: Path,
    should_crop: bool,
    class_remapping: dict
):
    """Обрабатывает и объединяет один split из исходного датасета."""
    dataset_name = source_dataset_dir.name
    print(f"--- Processing split: {dataset_name}/{split_name} ---")

    source_img_dir = source_dataset_dir / split_name / "images"
    source_label_dir = source_dataset_dir / split_name / "labels"

    image_paths = get_image_files(source_img_dir)
    if not image_paths:
        return

    for img_path in tqdm(image_paths, desc=f"Processing {dataset_name}/{split_name}"):
        label_path = source_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        img_height, img_width, _ = image.shape

        # Чтение и парсинг аннотаций
        lines = label_path.read_text().strip().split('\n')
        if not lines or not lines[0]:
            continue

        bboxes_data = []
        highest_y = img_height

        for line in lines:
            parts = line.split()
            local_class_id = int(parts[0])
            
            # Пропускаем класс, если его нет в глобальной карте
            if local_class_id not in class_remapping:
                continue
            
            global_class_id = class_remapping[local_class_id]
            coords = np.array([float(c) for c in parts[1:]])

            # Определяем тип аннотации
            is_segmentation = len(coords) > 4
            
            if is_segmentation:
                polygon = coords.reshape(-1, 2)
                bbox = convert_polygon_to_bbox(polygon.copy(), img_width, img_height)
                current_highest_y = np.min(polygon[:, 1] * img_height)
            else:
                center_x, center_y, box_width, box_height = coords
                bbox = (center_x, center_y, box_width, box_height)
                current_highest_y = (center_y - box_height / 2) * img_height

            if current_highest_y < highest_y:
                highest_y = current_highest_y
            
            bboxes_data.append((global_class_id, bbox))

        # Обрезка изображения при необходимости
        crop_y = 0
        new_img_height = img_height
        processed_image = image

        if should_crop:
            crop_y = max(0, int(highest_y - (img_height * CROP_MARGIN)))
            processed_image = image[crop_y:img_height, :]
            new_img_height, _, _ = processed_image.shape
            
            if new_img_height == 0:
                continue

        # Корректировка координат bbox после обрезки
        new_labels = []
        
        for global_class_id, (center_x, center_y, box_width, box_height) in bboxes_data:
            if should_crop:
                abs_y_center = center_y * img_height
                abs_y_height = box_height * img_height
                abs_y_min = abs_y_center - (abs_y_height / 2)
                abs_y_max = abs_y_center + (abs_y_height / 2)

                new_abs_y_min = max(0, abs_y_min - crop_y)
                new_abs_y_max = min(new_img_height, abs_y_max - crop_y)

                if new_abs_y_max <= new_abs_y_min:
                    continue

                new_abs_height = new_abs_y_max - new_abs_y_min
                new_abs_center_y = new_abs_y_min + (new_abs_height / 2)
                
                final_center_y = new_abs_center_y / new_img_height
                final_box_height = new_abs_height / new_img_height
                
                if final_box_height > 1e-4:
                    new_labels.append(
                        f"{global_class_id} {center_x} {final_center_y} "
                        f"{box_width} {final_box_height}"
                    )
            else:
                new_labels.append(
                    f"{global_class_id} {center_x} {center_y} "
                    f"{box_width} {box_height}"
                )

        # Сохранение результатов
        if new_labels:
            new_img_name = f"{dataset_name}_{img_path.name}"
            new_label_name = f"{dataset_name}_{img_path.stem}.txt"
            
            cv2.imwrite(str(dest_img_dir / new_img_name), processed_image)
            (dest_label_dir / new_label_name).write_text('\n'.join(new_labels))


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция для запуска обработки и объединения датасетов."""
    
    # Очистка и создание выходной директории
    if UNIFIED_OUTPUT_ROOT.exists():
        shutil.rmtree(UNIFIED_OUTPUT_ROOT)
    
    # Создание структуры директорий
    unified_train_images = UNIFIED_OUTPUT_ROOT / "train" / "images"
    unified_train_labels = UNIFIED_OUTPUT_ROOT / "train" / "labels"
    unified_test_images = UNIFIED_OUTPUT_ROOT / "test" / "images"
    unified_test_labels = UNIFIED_OUTPUT_ROOT / "test" / "labels"
    
    for path in [
        unified_train_images, unified_train_labels,
        unified_test_images, unified_test_labels
    ]:
        path.mkdir(parents=True)

    # Обработка каждого датасета
    for dataset_dir in INPUT_DATA_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Создание карты преобразования классов
        class_remapping = create_class_remapping(dataset_dir / "data.yaml")
        if not class_remapping:
            print(f"Skipping {dataset_name} due to missing class mapping.")
            continue

        # Определение необходимости обрезки
        should_crop = not (
            "cervica" in dataset_name.lower() or
            "cervical" in dataset_name.lower()
        )
        print(f"Cropping enabled: {should_crop}")

        # Обработка train и valid splits
        process_and_merge_split(
            dataset_dir, "train",
            unified_train_images, unified_train_labels,
            should_crop, class_remapping
        )
        
        process_and_merge_split(
            dataset_dir, "valid",
            unified_test_images, unified_test_labels,
            should_crop, class_remapping
        )

    print("\n" + "="*60)
    print("Script finished successfully!")
    print("="*60)


if __name__ == "__main__":
    main()