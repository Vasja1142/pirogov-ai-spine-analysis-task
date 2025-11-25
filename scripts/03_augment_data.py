"""
Скрипт аугментации данных.
Логика:
1. Resize: Короткая сторона приводится к MIN_SIZE (640).
2. Crop: Лишние пиксели (не кратные 32) обрезаются по центру.
3. Inversion: Создаются пары (Оригинал, Инверсия).
4. Augmentation: Изгиб, Шум.
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

INPUT_DATA_DIR = Path("data/02_processed")
OUTPUT_DATA_DIR = Path("data/03_augmented")

AUGMENTATIONS_PER_IMAGE = 2
MIN_SIZE = 640  # Короткая сторона будет приведена к этому размеру
STRIDE = 32     # Размеры должны быть кратны 32


# ============================================================================
# ФУНКЦИИ ТРАНСФОРМАЦИИ
# ============================================================================

def resize_and_crop_smart(image: np.ndarray, polygons: list) -> tuple:
    """
    Масштабирует изображение по короткой стороне до MIN_SIZE.
    Затем обрезает края, чтобы размеры были кратны STRIDE (32).
    Пересчитывает координаты полигонов.
    """
    h, w = image.shape[:2]
    
    # 1. Вычисляем масштаб
    scale = MIN_SIZE / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Ресайз
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Вычисляем размеры для кропа (кратные 32)
    crop_h = (new_h // STRIDE) * STRIDE
    crop_w = (new_w // STRIDE) * STRIDE
    
    # Если размеры уже идеальны, просто возвращаем
    if crop_h == new_h and crop_w == new_w:
        # Полигоны не меняются при простом ресайзе (они 0..1)
        return resized_img, polygons
    
    # 3. Вычисляем отступы для центрирования
    dy = (new_h - crop_h) // 2
    dx = (new_w - crop_w) // 2
    
    # Обрезка
    cropped_img = resized_img[dy:dy+crop_h, dx:dx+crop_w]
    
    # 4. Пересчет полигонов
    # При кропе меняется поле зрения, нужно сдвигать координаты
    new_polygons = []
    for poly in polygons:
        cls_id = poly[0]
        coords = np.array(poly[1:]).reshape(-1, 2)
        
        # Денормализация к размерам ПОСЛЕ РЕСАЙЗА (но до кропа)
        coords[:, 0] *= new_w
        coords[:, 1] *= new_h
        
        # Сдвиг (учитываем кроп)
        coords[:, 0] -= dx
        coords[:, 1] -= dy
        
        # Нормализация к НОВЫМ размерам (после кропа)
        coords[:, 0] /= crop_w
        coords[:, 1] /= crop_h
        
        # Обрезаем точки, ушедшие за пределы кадра
        coords = np.clip(coords, 0.0, 1.0)
        
        new_polygons.append([cls_id] + coords.flatten().tolist())
        
    return cropped_img, new_polygons


def apply_fast_cloud_noise(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    intensity = random.uniform(0.3, 0.6)
    random_noise = np.random.randn(h, w).astype(np.float32)
    blur_k = int(max(h, w) * random.uniform(0.5, 1.0))
    if blur_k % 2 == 0: blur_k += 1
    blurred_noise = cv2.GaussianBlur(random_noise, (blur_k, blur_k), 0)
    norm_noise = cv2.normalize(blurred_noise, None, -1, 1, cv2.NORM_MINMAX)
    multiplier = intensity + (norm_noise + 1) * (1.04 - intensity) / 2
    img_float = img.astype(np.float32)
    return np.clip(img_float * multiplier, 0, 255).astype(np.uint8)


def apply_hose_bend(img: np.ndarray, polygons: list, curvature: float) -> tuple:
    if abs(curvature) < 1e-6:
        return img, polygons
    h, w = img.shape[:2]
    max_shift = w * curvature
    new_w = w + int(np.ceil(abs(max_shift)))
    x_offset = int(np.ceil(abs(max_shift))) if curvature < 0 else 0
    final_x_coords, final_y_coords = np.meshgrid(np.arange(new_w), np.arange(h))
    shift_map = max_shift * np.sin(np.pi * final_y_coords / h)
    map_x = (final_x_coords - x_offset - shift_map).astype(np.float32)
    map_y = final_y_coords.astype(np.float32)
    processed_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    new_polygons = []
    for poly in polygons:
        class_id = poly[0]
        coords = np.array(poly[1:]).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        point_shifts = max_shift * np.sin(np.pi * coords[:, 1] / h)
        coords[:, 0] += point_shifts + x_offset
        coords[:, 0] /= new_w
        coords[:, 1] /= h
        coords = np.clip(coords, 0.0, 1.0)
        new_polygons.append([class_id] + coords.flatten().tolist())
    return processed_img, new_polygons


# ============================================================================
# IO UTILS
# ============================================================================

def load_polygons(path: Path):
    if not path.exists(): return []
    with open(path, 'r') as f:
        lines = f.readlines()
    polys = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 4:
            polys.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return polys

def save_polygons(path: Path, polys: list):
    with open(path, 'w') as f:
        for p in polys:
            cls = int(p[0])
            coords = " ".join([f"{x:.6f}" for x in p[1:]])
            f.write(f"{cls} {coords}\n")

def save_image_and_inverted_copy(img, polys, base_name, img_dir, lbl_dir):
    # 1. Normal
    cv2.imwrite(str(img_dir / f"{base_name}.png"), img)
    save_polygons(lbl_dir / f"{base_name}.txt", polys)
    # 2. Inverted
    img_inv = cv2.bitwise_not(img)
    cv2.imwrite(str(img_dir / f"{base_name}_inv.png"), img_inv)
    save_polygons(lbl_dir / f"{base_name}_inv.txt", polys)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if OUTPUT_DATA_DIR.exists(): shutil.rmtree(OUTPUT_DATA_DIR)
    shutil.copytree(INPUT_DATA_DIR, OUTPUT_DATA_DIR)
    
    split = "train"
    img_dir = OUTPUT_DATA_DIR / split / "images"
    label_dir = OUTPUT_DATA_DIR / split / "labels"
    
    # Удаляем файлы, скопированные copytree, мы их пересоздадим с новыми именами
    # но сначала запомним исходники
    source_img_dir = INPUT_DATA_DIR / split / "images"
    source_lbl_dir = INPUT_DATA_DIR / split / "labels"
    source_paths = sorted(list(source_img_dir.glob("*.jpg")) + list(source_img_dir.glob("*.png")))
    
    # Очистка output папок train перед заполнением
    for f in img_dir.glob("*"): f.unlink()
    for f in label_dir.glob("*"): f.unlink()

    print(f"Augmenting {split} set (Resize min={MIN_SIZE} + Crop + Inversion)...")

    for img_path in tqdm(source_paths):
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        lbl_path = source_lbl_dir / f"{img_path.stem}.txt"
        polygons = load_polygons(lbl_path)
        
        # --- 1. ПРИМЕНЯЕМ РЕСАЙЗ + КРОП (Базовая трансформация) ---
        base_img, base_polys = resize_and_crop_smart(image, polygons)

        # Сохраняем базу
        save_image_and_inverted_copy(
            base_img, base_polys, 
            f"{img_path.stem}_base", 
            img_dir, label_dir
        )

        # --- 2. АУГМЕНТАЦИИ (на основе уже уменьшенной картинки) ---
        for i in range(AUGMENTATIONS_PER_IMAGE):
            cur_img = base_img.copy()
            cur_polys = [p.copy() for p in base_polys]
            
            # Изгиб
            if random.random() < 0.7:
                curve = random.uniform(-0.15, 0.15)
                cur_img, cur_polys = apply_hose_bend(cur_img, cur_polys, curve)
            
            # Шум
            if random.random() < 0.8:
                gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
                noised = apply_fast_cloud_noise(gray)
                cur_img = cv2.cvtColor(noised, cv2.COLOR_GRAY2BGR)
            
            save_image_and_inverted_copy(
                cur_img, cur_polys,
                f"{img_path.stem}_aug_{i}",
                img_dir, label_dir
            )

    print(f"Done. Images resized to min_dim={MIN_SIZE} with smart cropping.")

if __name__ == "__main__":
    main()