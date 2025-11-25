print("Инициализация скрипта... (импорт библиотек)") # <-- Отладка

import shutil
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
    from ultralytics import YOLO
    print("Библиотеки загружены успешно.")
except ImportError as e:
    print(f"ОШИБКА ИМПОРТА: {e}")
    sys.exit(1)

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

# Проверь этот путь! Если модель не здесь, скрипт напишет об этом.
MODEL_PATH = Path("data/05_runs/spine_segmentation_v1/weights/best.pt")

RAW_IMAGES_DIR = Path("data/01_raw/new_raw_images") 

# Куда сохранять результат
OUTPUT_DIR = Path("data/01_raw/new_annotated_batch")

# Параметры препроцессинга
MIN_SIZE = 640
STRIDE = 32
MEDIAN_KSIZE = 3
GAMMA = 0.5
CLAHE_CLIP = 3.0
CLAHE_GRID = (30, 30)

# Пороги
CONF_THRESHOLD = 0.5
MIN_OBJECTS = 5           
MAX_OBJECTS = 25  

# Упрощение полигонов (0.003 - оптимально)
SIMPLIFY_EPSILON = 0.003

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def preprocess_image(image: np.ndarray):
    """Resize -> Crop -> Normalize"""
    h, w = image.shape[:2]
    scale = MIN_SIZE / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    crop_h = (new_h // STRIDE) * STRIDE
    crop_w = (new_w // STRIDE) * STRIDE
    
    dy = (new_h - crop_h) // 2
    dx = (new_w - crop_w) // 2
    
    cropped = resized[dy:dy+crop_h, dx:dx+crop_w]
    
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped
        
    denoised = cv2.medianBlur(gray, MEDIAN_KSIZE)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    
    inv_gamma = 1.0 / GAMMA
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corr = cv2.LUT(denoised, table)
    
    processed = clahe.apply(gamma_corr)
    
    mean, std = cv2.meanStdDev(processed)
    if std[0, 0] > 1e-6:
        processed = (processed - mean[0, 0]) / std[0, 0]
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

def simplify_contour(coords_norm: np.ndarray, h: int, w: int, epsilon_factor: float) -> np.ndarray:
    """Упрощает полигон алгоритмом Рамера-Дугласа-Пекера."""
    coords_abs = coords_norm.copy()
    coords_abs[:, 0] *= w
    coords_abs[:, 1] *= h
    coords_abs = coords_abs.astype(np.int32)
    
    perimeter = cv2.arcLength(coords_abs, True)
    epsilon = epsilon_factor * perimeter
    
    approx = cv2.approxPolyDP(coords_abs, epsilon, True)
    
    approx_norm = approx.reshape(-1, 2).astype(np.float32)
    approx_norm[:, 0] /= w
    approx_norm[:, 1] /= h
    
    approx_norm = np.clip(approx_norm, 0.0, 1.0)
    
    return approx_norm

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Запуск основной функции main()...")

    if not MODEL_PATH.exists():
        print(f"ОШИБКА: Модель не найдена по пути: {MODEL_PATH}")
        print("Проверьте, где лежит best.pt после обучения.")
        return

    if not RAW_IMAGES_DIR.exists():
        print(f"ВНИМАНИЕ: Папка {RAW_IMAGES_DIR} не найдена.")
        print(f"Создаю папку {RAW_IMAGES_DIR}...")
        RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print("Пожалуйста, положите в эту папку новые рентгеновские снимки и запустите скрипт снова.")
        return

    images = list(RAW_IMAGES_DIR.glob("*"))
    # Фильтруем только картинки
    images = [p for p in images if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not images:
        print(f"В папке {RAW_IMAGES_DIR} нет изображений!")
        return

    # Очистка и создание папок вывода
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / "confident" / "images").mkdir(parents=True)
    (OUTPUT_DIR / "confident" / "labels").mkdir(parents=True)
    (OUTPUT_DIR / "review_needed" / "images").mkdir(parents=True)
    (OUTPUT_DIR / "review_needed" / "labels").mkdir(parents=True)

    print(f"Загрузка модели YOLO из {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print(f"Найдено {len(images)} изображений. Начинаю разметку...")

    stats = {"confident": 0, "review": 0}

    for img_path in tqdm(images):
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print(f"Не удалось прочитать {img_path.name}")
            continue
        
        try:
            processed_img = preprocess_image(raw_img)
        except Exception as e:
            print(f"Ошибка обработки {img_path.name}: {e}")
            continue

        h, w = processed_img.shape[:2]

        # 1. ПРЕДИКТ
        results = model(processed_img, retina_masks=True, verbose=False)
        result = results[0]
        
        # 2. ПРОВЕРКА НА ИНВЕРСИЮ
        should_try_inverted = False
        if result.masks is None:
            should_try_inverted = True
        else:
            mean_conf = np.mean(result.boxes.conf.cpu().numpy())
            count = len(result.masks)
            if count < 3 or mean_conf < 0.4:
                should_try_inverted = True

        if should_try_inverted:
            inverted_img = cv2.bitwise_not(processed_img)
            results_inv = model(inverted_img, retina_masks=True, verbose=False)
            result_inv = results_inv[0]
            
            old_count = 0 if result.masks is None else len(result.masks)
            new_count = 0 if result_inv.masks is None else len(result_inv.masks)
            new_conf = 0 if result_inv.boxes is None else np.mean(result_inv.boxes.conf.cpu().numpy())
            
            if new_count > old_count or (new_count == old_count and new_conf > 0.6):
                processed_img = inverted_img
                result = result_inv

        # 3. АНАЛИЗ РЕЗУЛЬТАТОВ
        masks = result.masks
        boxes = result.boxes
        
        category = "confident" # По умолчанию

        if masks is None or len(masks) == 0:
            category = "review_needed"
        else:
            polygons = masks.xyn 
            scores = boxes.conf.cpu().numpy()
            mean_conf = np.mean(scores)
            obj_count = len(polygons)

            if (mean_conf < CONF_THRESHOLD) or \
               (obj_count < MIN_OBJECTS) or \
               (obj_count > MAX_OBJECTS):
                category = "review_needed"

            label_file = OUTPUT_DIR / category / "labels" / f"{img_path.stem}.txt"
            
            # Запись меток с упрощением
            with open(label_file, 'w') as f:
                for poly_raw in polygons:
                    poly_simple = simplify_contour(poly_raw, h, w, SIMPLIFY_EPSILON)
                    
                    if len(poly_simple) < 3: continue

                    coords_str = " ".join([f"{coord:.6f}" for coord in poly_simple.flatten()])
                    f.write(f"0 {coords_str}\n")

        # Сохранение изображения
        save_img_path = OUTPUT_DIR / category / "images" / f"{img_path.stem}.png"
        cv2.imwrite(str(save_img_path), processed_img)
        
        stats["confident" if category == "confident" else "review"] += 1

    print("\n" + "="*50)
    print("Готово!")
    print(f"Confident images: {stats['confident']}")
    print(f"Review needed:    {stats['review']}")
    print(f"Результат лежит в: {OUTPUT_DIR.absolute()}")
    print("="*50)

# !!! ВОТ ЭТА ЧАСТЬ САМАЯ ВАЖНАЯ !!!
if __name__ == "__main__":
    main()