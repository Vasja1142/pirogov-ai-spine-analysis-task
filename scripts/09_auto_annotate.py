"""
Скрипт для автоматической аннотации изображений с использованием модели YOLOv8 Segmentation.

Выполняет следующие шаги:
1.  Предварительная обработка изображений (опционально) для улучшения качества.
2.  Использование обученной модели YOLOv8 для генерации масок сегментации.
3.  Постобработка масок: эрозия, упрощение полигонов и фильтрация выбросов.
4.  Сохранение обработанных изображений и сгенерированных аннотаций в формате COCO JSON.

Пример использования:
python 09_auto_annotate.py --input-dir data/raw_images --output-img-dir data/auto_labeled \
--model-path data/best.pt --conf-threshold 0.5 --erosion-size 2
"""

import argparse
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional, Set

# ============================================================================
# ⚙️ КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class PreprocessingConfig:
    """Параметры для предварительной обработки изображений."""
    use_preprocessing: bool = True
    bilat_d: int = 3
    bilat_sigma_color: int = 75
    bilat_sigma_space: int = 75
    cutoff_percent: float = 0.03
    sharpen_sigma: float = 5.0
    sharpen_amount: float = 0.5

@dataclass
class AnnotationConfig:
    """Параметры для процесса аннотации и постобработки масок."""
    conf_threshold: float = 0.4
    class_name: str = "object"
    erosion_size: int = 1  # 0 = выключено, >0 = размер ядра эрозии
    poly_epsilon_factor: float = 0.01
    area_max_ratio: float = 4.0  # Максимальное отношение площади к средней
    area_min_ratio: float = 0.25 # Минимальное отношение площади к средней
    dim_max_ratio: float = 5.0   # Максимальное отношение ширины/высоты к средней
    min_polygon_area_pixels: float = 50.0 # Минимальная площадь полигона в пикселях

# ============================================================================
# 🛠 ФУНКЦИИ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
# ============================================================================

def apply_preprocessing_pipeline(img: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Применяет пайплайн предварительной обработки к изображению (шумоподавление, контраст, резкость).
    """
    if not config.use_preprocessing:
        return img

    if img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
    else:
        l_channel = img
        a_channel, b_channel = None, None # Для Ч/Б изображений

    # 1. Bilateral Filter
    l_channel = cv2.bilateralFilter(
        l_channel, config.bilat_d, config.bilat_sigma_color, config.bilat_sigma_space
    )

    # 2. Robust Auto-Levels (растягивание гистограммы)
    l_float = l_channel.astype(np.float32)
    low_val = np.percentile(l_float, config.cutoff_percent)
    high_val = np.percentile(l_float, 100 - config.cutoff_percent)
    l_clipped = np.clip(l_float, low_val, high_val)

    if high_val > low_val:
        l_norm = (l_clipped - low_val) / (high_val - low_val) * 255.0
    else:
        l_norm = l_clipped - low_val
    l_channel = np.clip(l_norm, 0, 255).astype(np.uint8)

    # 3. Unsharp Mask (резкость)
    gaussian = cv2.GaussianBlur(l_channel, (0, 0), config.sharpen_sigma)
    l_channel = cv2.addWeighted(
        l_channel, 1.0 + config.sharpen_amount, gaussian, -config.sharpen_amount, 0
    )

    if a_channel is not None and b_channel is not None:
        lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return l_channel

# ============================================================================
# 📐 ФУНКЦИИ ОБРАБОТКИ МАСОК И ПОЛИГОНОВ
# ============================================================================

def process_mask_to_polygon(
    mask_float: np.ndarray,
    img_shape: Tuple[int, int],
    anno_config: AnnotationConfig,
    erosion_kernel: Optional[np.ndarray]
) -> Optional[List[float]]:
    """
    Обрабатывает бинарную маску: эрозия, поиск контуров, упрощение полигона.
    Возвращает список плоских координат полигона или None.
    """
    h, w = img_shape
    mask_uint8 = (mask_float > 0.5).astype(np.uint8) * 255

    if erosion_kernel is not None:
        mask_uint8 = cv2.erode(mask_uint8, erosion_kernel, iterations=1)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Выбираем самый большой контур (или можно обработать все)
    main_contour = max(contours, key=cv2.contourArea)

    if len(main_contour) < 3:
        return None

    peri = cv2.arcLength(main_contour, True)
    epsilon = anno_config.poly_epsilon_factor * peri
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    if len(approx) < 3:
        return None

    area = cv2.contourArea(approx)
    if area < anno_config.min_polygon_area_pixels:
        return None

    # Нормализация координат и возврат плоского списка
    segmentation_coords = approx.flatten().astype(float) / np.array([w, h]).repeat(len(approx))
    return segmentation_coords.tolist()


def filter_candidate_polygons(
    candidates: List[Dict[str, Any]], anno_config: AnnotationConfig
) -> List[Dict[str, Any]]:
    """
    Фильтрует список кандидатов полигонов на основе статистических свойств.
    """
    if not candidates:
        return []

    areas = [c["area"] for c in candidates]
    if not areas: # Проверка на пустой список после получения areas
        return []

    mean_area = np.mean(areas)
    filtered_candidates = []

    for cand in candidates:
        # Фильтрация по площади
        if cand["area"] > mean_area * anno_config.area_max_ratio or \
           cand["area"] < mean_area * anno_config.area_min_ratio:
            continue
        
        # Фильтрация по соотношению сторон (ширина/высота или высота/ширина)
        if cand["w"] == 0 or cand["h"] == 0: # Избегаем деления на ноль
            continue
        dim_ratio = max(cand["w"], cand["h"]) / min(cand["w"], cand["h"])
        if dim_ratio > anno_config.dim_max_ratio:
            continue
            
        filtered_candidates.append(cand)

    return filtered_candidates

# ============================================================================
# 🚀 ГЛАВНАЯ ФУНКЦИЯ И ПАРСИНГ АРГУМЕНТОВ
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Автоматическая аннотация изображений с помощью YOLOv8-seg.")
    
    # === Основные параметры ===
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Путь к входной директории с изображениями для аннотации."
    )
    parser.add_argument(
        "--output-img-dir", type=Path, default=Path("data/auto_labeled"),
        help="Путь к выходной директории для сохранения обработанных изображений."
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("data/annotations_smart.json"),
        help="Путь к выходному JSON файлу в формате COCO."
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("data/05_runs/spine_segmentation_v2/weights/best.pt"),
        help="Путь к файлу обученной модели YOLOv8 Segmentation (например, best.pt)."
    )

    # === Параметры аннотации ===
    parser.add_argument(
        "--conf-threshold", type=float, default=0.4,
        help="Порог уверенности для обнаружений модели (0.0-1.0)."
    )
    parser.add_argument(
        "--class-name", type=str, default="object",
        help="Имя класса для всех автоматически сгенерированных аннотаций."
    )
    parser.add_argument(
        "--erosion-size", type=int, default=1,
        help="Размер ядра эрозии для сужения масок (в пикселях). 0 = выключено."
    )
    parser.add_argument(
        "--poly-epsilon-factor", type=float, default=0.01,
        help="Коэффициент для упрощения полигонов (относительно периметра)."
    )
    parser.add_argument(
        "--min-polygon-area-pixels", type=float, default=50.0,
        help="Минимальная площадь полигона в пикселях для сохранения."
    )

    # === Параметры фильтрации выбросов ===
    parser.add_argument(
        "--area-max-ratio", type=float, default=4.0,
        help="Максимальное отношение площади объекта к средней площади в изображении."
    )
    parser.add_argument(
        "--area-min-ratio", type=float, default=0.25,
        help="Минимальное отношение площади объекта к средней площади в изображении."
    )
    parser.add_argument(
        "--dim-max-ratio", type=float, default=5.0,
        help="Максимальное отношение большей стороны bbox к меньшей (для фильтрации вытянутых объектов)."
    )

    # === Параметры предварительной обработки ===
    parser.add_argument(
        "--no-preprocessing", action="store_true",
        help="Отключить предварительную обработку изображений."
    )
    parser.add_argument(
        "--bilat-d", type=int, default=3,
        help="Диаметр окрестности для bilateralFilter."
    )
    parser.add_argument(
        "--bilat-sigma-color", type=int, default=75,
        help="SigmaColor для bilateralFilter."
    )
    parser.add_argument(
        "--bilat-sigma-space", type=int, default=75,
        help="SigmaSpace для bilateralFilter."
    )
    parser.add_argument(
        "--cutoff-percent", type=float, default=0.03,
        help="Процент отсечения для Robust Auto-Levels (0.0-1.0)."
    )
    parser.add_argument(
        "--sharpen-sigma", type=float, default=5.0,
        help="Sigma для GaussianBlur в Unsharp Mask."
    )
    parser.add_argument(
        "--sharpen-amount", type=float, default=0.5,
        help="Сила резкости для Unsharp Mask (0.0 и выше)."
    )

    args = parser.parse_args()

    # Инициализация конфигураций
    pre_config = PreprocessingConfig(
        use_preprocessing=not args.no_preprocessing,
        bilat_d=args.bilat_d,
        bilat_sigma_color=args.bilat_sigma_color,
        bilat_sigma_space=args.bilat_sigma_space,
        cutoff_percent=args.cutoff_percent,
        sharpen_sigma=args.sharpen_sigma,
        sharpen_amount=args.sharpen_amount,
    )
    anno_config = AnnotationConfig(
        conf_threshold=args.conf_threshold,
        class_name=args.class_name,
        erosion_size=args.erosion_size,
        poly_epsilon_factor=args.poly_epsilon_factor,
        area_max_ratio=args.area_max_ratio,
        area_min_ratio=args.area_min_ratio,
        dim_max_ratio=args.dim_max_ratio,
        min_polygon_area_pixels=args.min_polygon_area_pixels,
    )

    # === Проверки путей ===
    if not args.input_dir.is_dir():
        print(f"❌ Ошибка: Входная директория не найдена: {args.input_dir}")
        return
    if not args.model_path.is_file():
        print(f"❌ Ошибка: Файл модели не найден: {args.model_path}")
        return

    # Очистка и создание выходных директорий
    if args.output_img_dir.exists():
        shutil.rmtree(args.output_img_dir)
    args.output_img_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка модели
    print(f"[*] Загрузка модели: {args.model_path}")
    model = YOLO(str(args.model_path))

    coco_output: Dict[str, Any] = {
        "info": {"description": "Auto-labeling with YOLOv8 Segmentation and Post-processing"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": anno_config.class_name, "supercategory": "object"}],
    }

    image_files: List[Path] = sorted(list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.png")))
    if not image_files:
        print(f"⚠️ В директории {args.input_dir} не найдено изображений для обработки.")
        return

    print(f"Обработка {len(image_files)} изображений...")
    print(f"Конфигурация: Confidence={anno_config.conf_threshold}, Erosion={anno_config.erosion_size}px")

    ann_id: int = 1
    # Ядро для эрозии, если EROSION_SIZE > 0
    erosion_kernel: Optional[np.ndarray] = (
        np.ones((anno_config.erosion_size * 2 + 1, anno_config.erosion_size * 2 + 1), np.uint8)
        if anno_config.erosion_size > 0
        else None
    )

    for img_id, img_path in enumerate(tqdm(image_files, desc="Аннотирование"), start=1):
        original_img: Optional[np.ndarray] = cv2.imread(str(img_path))
        if original_img is None:
            print(f"  [Предупреждение] Не удалось прочитать изображение: {img_path.name}")
            continue

        # Предварительная обработка
        processed_img: np.ndarray = apply_preprocessing_pipeline(original_img, pre_config)
        h, w = processed_img.shape[:2]

        # Сохранение обработанного изображения
        cv2.imwrite(str(args.output_img_dir / img_path.name), processed_img)

        coco_output["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
        )

        # Предсказание модели
        results = model.predict(
            processed_img, conf=anno_config.conf_threshold, retina_masks=True, verbose=False
        )
        result = results[0]

        if result.masks is None or result.masks.data.numel() == 0: # Проверка на пустые маски
            continue

        masks_data: np.ndarray = result.masks.data.cpu().numpy()

        # Изменение размера масок, если они не совпадают с размерами изображения
        if masks_data.shape[1:] != (h, w):
            masks_data_resized: List[np.ndarray] = []
            for m in masks_data:
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                masks_data_resized.append(m_resized)
            masks_data = np.array(masks_data_resized)

        candidate_polygons: List[Dict[str, Any]] = []

        for mask_float in masks_data:
            segmentation = process_mask_to_polygon(
                mask_float, (h, w), anno_config, erosion_kernel
            )
            if segmentation:
                # Получаем bbox из полигона для COCO
                poly_np = np.array(segmentation).reshape(-1, 2)
                # Денормализуем для расчета bbox в пикселях
                poly_px = poly_np * np.array([w, h])
                x, y, rect_w, rect_h = cv2.boundingRect(poly_px.astype(np.int32))
                bbox = [float(x), float(y), float(rect_w), float(rect_h)]
                area = cv2.contourArea(poly_px.astype(np.int32))

                candidate_polygons.append({
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "w": rect_w,
                    "h": rect_h,
                })
        
        # Применяем фильтрацию к кандидатам
        filtered_polygons = filter_candidate_polygons(candidate_polygons, anno_config)

        for cand in filtered_polygons:
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [cand["segmentation"]], # COCO ожидает список списков
                "area": cand["area"],
                "bbox": cand["bbox"],
                "iscrowd": 0,
            })
            ann_id += 1

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Автоматическая аннотация завершена!")
    print(f"  Обработанные изображения сохранены в: {args.output_img_dir.resolve()}")
    print(f"  Аннотации в формате COCO JSON сохранены в: {args.output_json.resolve()}")

if __name__ == "__main__":
    main()