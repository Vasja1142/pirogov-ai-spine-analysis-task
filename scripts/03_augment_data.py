"""
Скрипт для аугментации набора данных изображений.

Основные шаги:
1.  Копирует существующий обработанный набор данных в новую директорию.
2.  Для каждого изображения в обучающем наборе (`train`) применяет серию аугментаций.
3.  Сохраняет как оригинальные, так и аугментированные изображения и их метки.
4.  Аугментации включают:
    - Геометрические искажения (изгиб позвоночника).
    - Текстурные изменения (эластичная трансформация, размытие).
    - Наложение шума (облачный шум).
    - Инверсия цвета.
5.  Включает проверку валидности полигонов после трансформаций.
"""
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from typing import List, Dict, Tuple

# ============================================================================ 
# КОНФИГУРАЦИЯ
# ============================================================================ 
INPUT_DATA_DIR = Path("data/02_processed")
OUTPUT_DATA_DIR = Path("data/03_augmented")
# Количество аугментаций, генерируемых для каждого исходного изображения
AUGMENTATIONS_PER_IMAGE: int = 5
# Шаг для обрезки изображений, чтобы их размеры были кратны этому числу
STRIDE: int = 32

# ============================================================================ 
# 1. ПРОВЕРКА ВАЛИДНОСТИ ПОЛИГОНОВ
# ============================================================================ 

def calculate_polygon_area(coords: np.ndarray) -> float:
    """Вычисляет площадь полигона по его координатам (формула шнурков)."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def is_valid_polygon(coords: np.ndarray) -> bool:
    """
    Проверяет, является ли полигон валидным после аугментации.

    Критерии невалидности:
    - Менее 3-х вершин.
    - Слишком маленькая площадь.
    - Чрезмерная высота (более 60% от высоты изображения).
    - Слишком много точек "прилипло" к границам изображения.
    """
    if len(coords) < 3:
        return False

    if calculate_polygon_area(coords) < 0.001:
        return False

    if (coords[:, 1].max() - coords[:, 1].min()) > 0.6:
        return False

    edge_tolerance = 1e-3
    on_edge = (
        (coords[:, 0] < edge_tolerance) | (coords[:, 0] > 1 - edge_tolerance) |
        (coords[:, 1] < edge_tolerance) | (coords[:, 1] > 1 - edge_tolerance)
    )
    if np.sum(on_edge) / len(coords) > 0.3:
        return False

    return True

# ============================================================================ 
# 2. ФУНКЦИИ АУГМЕНТАЦИИ
# ============================================================================ 

def apply_advanced_spine_curve(
    img: np.ndarray, polygons: List[list]
) -> Tuple[np.ndarray, List[list]]:
    """Применяет S-образный изгиб к изображению и пересчитывает координаты полигонов."""
    h, w = img.shape[:2]
    amplitude = w * random.uniform(0.01, 0.05)
    periods = random.uniform(0.6, 1.2)
    phase = random.uniform(0, 2 * np.pi)
    direction = random.choice([-1, 1])

    max_shift = abs(amplitude)
    new_w = w + int(np.ceil(max_shift * 2))
    x_offset = int(np.ceil(max_shift))

    map_y, map_x = np.indices((h, new_w), dtype=np.float32)
    y_normalized = map_y / h
    shift = direction * amplitude * np.sin(2 * np.pi * periods * y_normalized + phase)

    src_map_x = map_x - x_offset - shift
    curved_img = cv2.remap(
        img, src_map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    new_polygons = []
    for poly in polygons:
        cls_id, coords_flat = poly[0], poly[1:]
        coords = np.array(coords_flat).reshape(-1, 2)
        px_x, px_y = coords[:, 0] * w, coords[:, 1] * h
        point_shifts = direction * amplitude * np.sin(2 * np.pi * periods * (px_y / h) + phase)
        new_x = (px_x + x_offset + point_shifts) / new_w
        new_y = px_y / h
        final_coords = np.clip(np.column_stack((new_x, new_y)), 0.0, 1.0)

        if is_valid_polygon(final_coords):
            new_polygons.append([cls_id] + final_coords.flatten().tolist())

    return curved_img, new_polygons

def get_texture_augs() -> A.Compose:
    """Возвращает композицию текстурных аугментаций от Albumentations."""
    return A.Compose([
        A.OneOf([
            A.ElasticTransform(alpha=600, sigma=100, p=0.5),
            A.ElasticTransform(alpha=300, sigma=50, p=0.5),
            A.ElasticTransform(alpha=50, sigma=20, p=0.5),
        ], p=1.0),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ], p=0.2),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def apply_albumentations(
    img: np.ndarray, polygons: List[list]
) -> Tuple[np.ndarray, List[list]]:
    """Применяет текстурные аугментации и корректно обрабатывает полигоны."""
    h, w = img.shape[:2]
    all_keypoints, poly_slices = [], []
    current_idx = 0

    for poly in polygons:
        cls_id, coords_flat = poly[0], poly[1:]
        coords = np.array(coords_flat).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        points_list = coords.tolist()
        all_keypoints.extend(points_list)
        poly_slices.append({"start": current_idx, "end": current_idx + len(points_list), "cls": cls_id})
        current_idx += len(points_list)

    if not all_keypoints:
        return img, polygons

    transformed = get_texture_augs()(image=img, keypoints=all_keypoints)
    new_img, new_kps = transformed["image"], transformed["keypoints"]

    if len(new_kps) != len(all_keypoints):
        print("  [Предупреждение] Потеря точек при аугментации. Пропуск трансформации.")
        return img, polygons

    final_polygons = []
    for s in poly_slices:
        poly_pts = new_kps[s["start"] : s["end"]]
        processed_pts = [
            val for x, y in poly_pts for val in (np.clip(x / w, 0, 1), np.clip(y / h, 0, 1))
        ]
        pts_check = np.array(processed_pts).reshape(-1, 2)
        if is_valid_polygon(pts_check):
            final_polygons.append([s["cls"]] + processed_pts)

    return new_img, final_polygons

def apply_fast_cloud_noise(img: np.ndarray) -> np.ndarray:
    """Накладывает на изображение процедурный облачный шум."""
    h, w = img.shape[:2]
    intensity = random.uniform(0.4, 0.7)
    random_noise = np.random.randn(h, w).astype(np.float32)
    blur_scale = random.uniform(0.7, 1.2)
    ksize = int(max(h, w) * blur_scale)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize

    blurred_noise = cv2.GaussianBlur(random_noise, (ksize, ksize), 0)
    norm_noise = cv2.normalize(blurred_noise, None, -1, 1, cv2.NORM_MINMAX)
    multiplier = intensity + (norm_noise + 1) * (1.04 - intensity) / 2

    img_float = img.astype(np.float32)
    if img.ndim == 3:
        multiplier = np.expand_dims(multiplier, axis=-1)

    return np.clip(img_float * multiplier, 0, 255).astype(np.uint8)

# ============================================================================ 
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================ 

def resize_and_crop_smart(
    image: np.ndarray, polygons: List[list]
) -> Tuple[np.ndarray, List[list]]:
    """Обрезает изображение до размеров, кратных STRIDE."""
    h, w = image.shape[:2]
    crop_h, crop_w = (h // STRIDE) * STRIDE, (w // STRIDE) * STRIDE
    if crop_h == h and crop_w == w:
        return image, polygons

    dy, dx = (h - crop_h) // 2, (w - crop_w) // 2
    cropped_img = image[dy:dy + crop_h, dx:dx + crop_w]

    new_polygons = []
    for poly in polygons:
        cls_id, coords_flat = poly[0], poly[1:]
        coords = np.array(coords_flat).reshape(-1, 2)
        coords[:, 0] = (coords[:, 0] * w - dx) / crop_w
        coords[:, 1] = (coords[:, 1] * h - dy) / crop_h
        coords = np.clip(coords, 0.0, 1.0)
        if is_valid_polygon(coords):
            new_polygons.append([cls_id] + coords.flatten().tolist())

    return cropped_img, new_polygons

def load_polygons(path: Path) -> List[List[float]]:
    """Загружает полигоны из текстового файла."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [
            [int(p[0])] + [float(x) for x in p[1:]]
            for line in f if len(p := line.split()) > 4
        ]

def save_result(img: np.ndarray, polys: List[list], name: str, img_d: Path, lbl_d: Path):
    """Сохраняет изображение и файл меток."""
    cv2.imwrite(str(img_d / f"{name}.jpg"), img)
    with open(lbl_d / f"{name}.txt", "w") as f:
        for p in polys:
            f.write(f"{int(p[0])} {" ".join(f'{x:.6f}' for x in p[1:])}\n")

# ============================================================================ 
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================ 

def main():
    """Основной цикл выполнения скрипта аугментации."""
    if OUTPUT_DATA_DIR.exists():
        shutil.rmtree(OUTPUT_DATA_DIR)
    shutil.copytree(INPUT_DATA_DIR, OUTPUT_DATA_DIR, dirs_exist_ok=True)

    img_dir = OUTPUT_DATA_DIR / "images" / "train"
    label_dir = OUTPUT_DATA_DIR / "labels" / "train"
    src_img_dir = INPUT_DATA_DIR / "images" / "train"
    src_lbl_dir = INPUT_DATA_DIR / "labels" / "train"

    if not src_img_dir.is_dir():
        print(f"ОШИБКА: Исходная директория не найдена: {src_img_dir}")
        return

    # Очистка целевых папок от скопированных файлов
    for f in img_dir.glob("*"): f.unlink()
    for f in label_dir.glob("*"): f.unlink()

    images = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
    total_new = len(images) * (1 + AUGMENTATIONS_PER_IMAGE)
    print(f"🧬 Генерация аугментаций: {len(images)} исходных -> ~{total_new} результирующих.")

    for img_path in tqdm(images, desc="Аугментация"):
        image = cv2.imread(str(img_path))
        if image is None: continue

        polygons = load_polygons(src_lbl_dir / f"{img_path.stem}.txt")
        base_img, base_polys = resize_and_crop_smart(image, polygons)
        save_result(base_img, base_polys, f"{img_path.stem}_base", img_dir, label_dir)

        for i in range(AUGMENTATIONS_PER_IMAGE):
            cur_img, cur_polys = base_img.copy(), [p.copy() for p in base_polys]

            if random.random() < 0.8:
                cur_img, cur_polys = apply_advanced_spine_curve(cur_img, cur_polys)
            cur_img, cur_polys = apply_albumentations(cur_img, cur_polys)
            if random.random() < 0.5:
                cur_img = apply_fast_cloud_noise(cur_img)

            suffix = "inv" if random.random() < 0.5 else "norm"
            if suffix == "inv":
                cur_img = cv2.bitwise_not(cur_img)

            if cur_polys:
                save_result(cur_img, cur_polys, f"{img_path.stem}_aug{i}_{suffix}", img_dir, label_dir)

    print("✅ Аугментация завершена.")

if __name__ == "__main__":
    main()