"""
Скрипт для предварительной аугментации и подготовки объединенного набора данных.

Гибридный подход (Финальная версия):
- Сочетает надежность ручного изгиба с удобством Albumentations
- Последовательно применяет трансформации с разной вероятностью:
  1. Эластичная деформация (через пайплайн Albumentations)
  2. Вертикальный изгиб (через ручную функцию)
- Позволяет получить изображения с изгибом, эластиком или обоими эффектами
- Корректно фильтрует метки классов вместе с отброшенными рамками
"""

import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

INPUT_DATA_DIR = Path("data/02_processed")
OUTPUT_DATA_DIR = Path("data/03_augmented")
AUGMENTATIONS_PER_IMAGE = 7
MAX_SIZE = 1000
BORDER_MODE_CONSTANT = cv2.BORDER_CONSTANT
BORDER_MODE_REFLECT = cv2.BORDER_REFLECT_101


# ============================================================================
# ФУНКЦИИ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
# ============================================================================

def apply_fast_cloud_noise(img: np.ndarray) -> np.ndarray:
    """Применяет облачный шум к изображению для имитации текстуры."""
    h, w = img.shape[:2]
    intensity = random.uniform(0.4, 0.7)
    
    # Генерация случайного шума
    random_noise = np.random.randn(h, w).astype(np.float32)
    
    # Размытие шума для создания облачного эффекта
    blur_scale_factor = random.uniform(0.4, 1.6)
    blur_kernel_size = int(max(h, w) * blur_scale_factor)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    blurred_noise = cv2.GaussianBlur(
        random_noise,
        (blur_kernel_size, blur_kernel_size),
        0
    )
    
    # Нормализация и применение к изображению
    norm_noise = cv2.normalize(blurred_noise, None, -1, 1, cv2.NORM_MINMAX)
    multiplier = intensity + (norm_noise + 1) * (1.04 - intensity) / 2
    
    img_float = img.astype(np.float32)
    multiplied_float = img_float * multiplier
    
    return np.clip(multiplied_float, 0, 255).astype(np.uint8)


def apply_hose_bend_to_image(img: np.ndarray, curvature: float) -> np.ndarray:
    """
    Применяет вертикальный изгиб к изображению.
    
    Args:
        img: Входное изображение
        curvature: Коэффициент кривизны изгиба
    
    Returns:
        Изогнутое изображение
    """
    if abs(curvature) < 1e-6:
        return img
    
    h, w = img.shape[:2]
    max_shift = w * curvature
    new_w = w + int(np.ceil(abs(max_shift)))
    x_offset = int(np.ceil(abs(max_shift))) if curvature < 0 else 0
    
    # Создание карт деформации
    final_x_coords, final_y_coords = np.meshgrid(
        np.arange(new_w),
        np.arange(h)
    )
    
    shift = max_shift * np.sin(np.pi * final_y_coords / h)
    map_x = (final_x_coords - x_offset - shift).astype(np.float32)
    map_y = final_y_coords.astype(np.float32)
    
    return cv2.remap(
        src=img,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=BORDER_MODE_CONSTANT,
        borderValue=(0, 0, 0)
    )


def apply_hose_bend_to_bboxes_and_labels(
    bboxes: list,
    class_labels: list,
    h: int,
    w: int,
    curvature: float
) -> tuple[list, list]:
    """
    Применяет трансформацию изгиба к bounding boxes и их меткам.
    
    Args:
        bboxes: Список bbox в формате YOLO [x_c, y_c, width, height]
        class_labels: Список меток классов
        h: Высота изображения
        w: Ширина изображения
        curvature: Коэффициент кривизны
    
    Returns:
        Кортеж (новые_bbox, новые_метки)
    """
    if abs(curvature) < 1e-6:
        return bboxes, class_labels
    
    new_bboxes, new_labels = [], []
    max_shift = w * curvature
    new_w = w + int(np.ceil(abs(max_shift)))
    x_offset = int(np.ceil(abs(max_shift))) if curvature < 0 else 0

    for i, bbox in enumerate(bboxes):
        x_c, y_c, width, height = bbox
        
        # Денормализация координат
        x_min, y_min = (x_c - width / 2) * w, (y_c - height / 2) * h
        x_max, y_max = (x_c + width / 2) * w, (y_c + height / 2) * h
        
        # Трансформация углов bbox
        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        
        shift = max_shift * np.sin(np.pi * corners[:, 1] / h)
        corners[:, 0] += shift
        corners[:, 0] += x_offset
        
        # Вычисление новых границ
        new_x_min, new_x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
        new_y_min, new_y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
        
        # Фильтрация невалидных bbox
        if new_x_max <= 0 or new_x_min >= new_w or new_x_min >= new_x_max:
            continue
        
        # Нормализация обратно в формат YOLO
        new_x_c = (new_x_min + new_x_max) / (2 * new_w)
        new_y_c = (new_y_min + new_y_max) / (2 * h)
        new_width = (new_x_max - new_x_min) / new_w
        new_height = (new_y_max - new_y_min) / h
        
        # Проверка валидности
        if (new_width > 0 and new_height > 0 and
            0 <= new_x_c <= 1 and 0 <= new_y_c <= 1):
            new_bboxes.append([new_x_c, new_y_c, new_width, new_height])
            new_labels.append(class_labels[i])
    
    return new_bboxes, new_labels


# ============================================================================
# ФУНКЦИИ РАБОТЫ С АННОТАЦИЯМИ
# ============================================================================

def load_yolo_annotations(label_path: Path) -> tuple[list, list]:
    """Загружает аннотации YOLO из файла."""
    if not label_path.exists():
        return [], []
    
    annotations = np.loadtxt(str(label_path), delimiter=' ', ndmin=2)
    bboxes = annotations[:, 1:].tolist()
    class_labels = annotations[:, 0].astype(int).tolist()
    
    return bboxes, class_labels


def save_yolo_annotations(
    label_path: Path,
    bboxes: list,
    class_labels: list
):
    """Сохраняет аннотации YOLO в файл."""
    if len(bboxes) == 0:
        label_path.touch()
        return
    
    labels = np.hstack((
        np.array(class_labels).reshape(-1, 1),
        np.array(bboxes)
    ))
    
    np.savetxt(str(label_path), labels, fmt='%d %f %f %f %f')


# ============================================================================
# КЛАССЫ И ПАЙПЛАЙНЫ АУГМЕНТАЦИИ
# ============================================================================

class ElasticTransformWithRandomSigma(A.ElasticTransform):
    """Эластичная трансформация с рандомизированным sigma."""
    
    def __init__(self, *args, sigma_limit=(20, 50), **kwargs):
        super().__init__(*args, sigma=50, **kwargs)
        self.sigma_limit = sigma_limit
    
    def __call__(self, *args, **kwargs):
        self.sigma = random.uniform(self.sigma_limit[0], self.sigma_limit[1])
        return super().__call__(*args, **kwargs)


# Пайплайн для изменения размера
resize_transform = A.Compose([
    A.LongestMaxSize(max_size=MAX_SIZE, interpolation=cv2.INTER_AREA)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Пайплайн эластичной деформации
elastic_transform_pipeline = A.Compose([
    A.OneOf([
        ElasticTransformWithRandomSigma(
            alpha=600,
            sigma_limit=(50, 100),
            border_mode=BORDER_MODE_REFLECT,
            p=0.4
        ),
        ElasticTransformWithRandomSigma(
            alpha=300,
            sigma_limit=(20, 50),
            border_mode=BORDER_MODE_REFLECT,
            p=0.4
        ),
        ElasticTransformWithRandomSigma(
            alpha=50,
            sigma_limit=(5, 20),
            border_mode=BORDER_MODE_REFLECT,
            p=0.2
        ),
    ], p=1.0),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.1
))


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция для запуска аугментации датасета."""
    
    # Подготовка выходной директории
    if OUTPUT_DATA_DIR.exists():
        shutil.rmtree(OUTPUT_DATA_DIR)
    shutil.copytree(INPUT_DATA_DIR, OUTPUT_DATA_DIR)
    
    # Обработка каждого split
    for split in ["train", "valid", "test"]:
        split_path = OUTPUT_DATA_DIR / split
        if not split_path.exists():
            continue
        
        img_dir = split_path / "images"
        label_dir = split_path / "labels"
        
        # Получение списка оригинальных изображений
        original_image_paths = sorted(
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png"))
        )
        
        if not original_image_paths:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")
        
        # Обработка каждого изображения
        for img_path in tqdm(
            original_image_paths,
            desc=f"Augmenting {split} images"
        ):
            image_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_color is None:
                continue
            
            # Загрузка аннотаций
            label_path = label_dir / f"{img_path.stem}.txt"
            bboxes, class_labels = load_yolo_annotations(label_path)
            
            if not class_labels:
                continue

            # Изменение размера при необходимости
            h, w, _ = image_color.shape
            if h > MAX_SIZE or w > MAX_SIZE:
                resized_data = resize_transform(
                    image=image_color,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                base_image = resized_data['image']
                base_bboxes = resized_data['bboxes']
                base_labels = resized_data['class_labels']
            else:
                base_image = image_color
                base_bboxes = bboxes
                base_labels = class_labels

            # Генерация аугментированных вариантов
            for i in range(AUGMENTATIONS_PER_IMAGE):
                try:
                    # Создаем рабочие копии
                    current_image = base_image.copy()
                    current_bboxes = [b.copy() for b in base_bboxes]
                    current_labels = base_labels.copy()
                    
                    was_transformed = False

                    # ЭТАП 1: Эластичная деформация (Albumentations)
                    if random.random() < 0.8:
                        transformed = elastic_transform_pipeline(
                            image=current_image,
                            bboxes=current_bboxes,
                            class_labels=current_labels
                        )
                        current_image = transformed['image']
                        current_bboxes = transformed['bboxes']
                        current_labels = transformed['class_labels']
                        was_transformed = True

                    # ЭТАП 2: Вертикальный изгиб (ручная функция)
                    if random.random() < 0.8:
                        h_before_bend, w_before_bend, _ = current_image.shape
                        curvature = random.uniform(-0.2, 0.2)
                        
                        current_image = apply_hose_bend_to_image(
                            current_image,
                            curvature
                        )
                        current_bboxes, current_labels = \
                            apply_hose_bend_to_bboxes_and_labels(
                                current_bboxes,
                                current_labels,
                                h_before_bend,
                                w_before_bend,
                                curvature
                            )
                        was_transformed = True

                    # Пропускаем если не было трансформаций
                    if not was_transformed:
                        continue
                    
                    if len(current_bboxes) == 0:
                        continue
                    
                    # ФИНАЛЬНЫЙ ЭТАП: Шум и сохранение
                    image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                    final_image = apply_fast_cloud_noise(image_gray)
                    
                    # Сохранение результата
                    aug_stem = f"{img_path.stem}_aug_{i}"
                    aug_img_path = img_dir / f"{aug_stem}.png"
                    aug_label_path = label_dir / f"{aug_stem}.txt"
                    
                    cv2.imwrite(str(aug_img_path), final_image)
                    save_yolo_annotations(
                        aug_label_path,
                        current_bboxes,
                        current_labels
                    )

                except Exception as e:
                    tqdm.write(f"\nОшибка при аугментации {img_path.name}: {e}")

    print("\n" + "="*60)
    print("Скрипт аугментации успешно завершен!")
    print("="*60)


if __name__ == "__main__":
    main()