"""
Скрипт для визуальной проверки корректности меток на изображениях.

Функциональность:
1. Читает data.yaml для получения списка имен классов
2. Выбирает случайное изображение из указанного каталога
3. Находит соответствующий файл меток
4. Рисует все bounding boxes с подписями классов
5. Сохраняет результат в data/05_verification_runs
"""

import random
from pathlib import Path

import cv2
import numpy as np
import yaml


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Путь к основному data.yaml с глобальными классами
DATA_YAML_PATH = Path("data.yaml")

# Каталог с изображениями для проверки
IMAGES_DIR = Path("data/03_augmented_MANUAL/train/images")

# Каталог для сохранения результатов проверки
OUTPUT_DIR = Path("data/05_verification_runs")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция для запуска проверки."""
    
    # Проверка существования data.yaml
    if not DATA_YAML_PATH.exists():
        print(f"Error: {DATA_YAML_PATH} not found.")
        return
    
    # Чтение списка классов
    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    
    # Проверка существования директории с изображениями
    if not IMAGES_DIR.exists():
        print(f"Error: Images directory {IMAGES_DIR} not found.")
        return

    # Получение списка изображений
    image_paths = (
        list(IMAGES_DIR.glob("*.jpg")) +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )
    
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}.")
        return

    # Выбор случайного изображения
    random_image_path = random.choice(image_paths)
    label_path = IMAGES_DIR.parent / "labels" / f"{random_image_path.stem}.txt"

    print(f"Verifying image: {random_image_path.name}")

    # Проверка существования файла меток
    if not label_path.exists():
        print(f"Label file not found for {random_image_path.name}")
        return

    # Чтение изображения и меток
    image = cv2.imread(str(random_image_path))
    img_height, img_width, _ = image.shape
    
    annotations = np.loadtxt(str(label_path), delimiter=' ', ndmin=2)
    
    # Генерация цветов для каждого класса
    colors = [
        tuple(np.random.randint(100, 256, 3).tolist())
        for _ in range(len(class_names))
    ]

    # Отрисовка каждой аннотации
    for ann in annotations:
        class_id = int(ann[0])
        center_x, center_y, box_width, box_height = ann[1:]

        # Денормализация координат
        abs_center_x = int(center_x * img_width)
        abs_center_y = int(center_y * img_height)
        abs_width = int(box_width * img_width)
        abs_height = int(box_height * img_height)

        # Вычисление координат углов bbox
        x_min = int(abs_center_x - abs_width / 2)
        y_min = int(abs_center_y - abs_height / 2)
        x_max = int(abs_center_x + abs_width / 2)
        y_max = int(abs_center_y + abs_height / 2)

        # Рисование bounding box
        class_name = class_names[class_id]
        color = colors[class_id]
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Подготовка и отрисовка текста
        label = f"{class_id}: {class_name}"
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )
        
        # Фон для текста
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - 5),
            (x_min + text_width, y_min),
            color,
            -1
        )
        
        # Текст
        cv2.putText(
            image,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # Сохранение результата
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"verified_{random_image_path.name}"
    cv2.imwrite(str(output_path), image)
    
    print(f"Verification image saved to: {output_path}")


if __name__ == "__main__":
    main()