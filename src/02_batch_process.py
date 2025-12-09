import os
import cv2
import numpy as np
import shutil
import yaml  # Добавили для редактирования конфига
from glob import glob
from tqdm import tqdm

# --- НАСТРОЙКИ ФИЛЬТРОВ ---
CONFIG = {
    # 1. Progressive Filler
    "use_fill": True,
    "fill_max_k": 6,
    # 2. Multipass CLAHE
    "use_clahe": True,
    "clahe_clip": 4.50,
    "clahe_grid": 24,
    "clahe_passes": 8,
    # 3. Bilateral (Float)
    "use_bilat": False,
    "bilat_d": 20,  # Изменено на 20
    "bilat_sc": 5,
    "bilat_ss": 0.90,  # Изменено на 0.90
    # 4. NLM (Cleaner)
    "use_nlm": False,
    "nlm_h": 4,
    "nlm_t": 7,
    "nlm_s": 7,
    # 5. Резкость
    "use_usm": False,
    "usm_amt": 2.20,
    "usm_sigma": 1.00,
    "usm_smart_blur": 3.98,
    "use_lap": False,
    "lap_str": 0.40,  # Изменено на 5.00
    "lap_blur": 0.00,  # Изменено на 4.00
    # 6. SSAA
    "use_ssaa": False,  # Выключено
    "ssaa_factor": 2.0,
    "ssaa_smooth": 1.5,
}

DIRS = {"input": "data/02_processed", "output": "data/03_enhanced"}

# --- ФУНКЦИИ ОБРАБОТКИ ---


def apply_progressive_hole_filling(img, max_k=5):
    result = img.copy()
    for k in range(3, max_k + 1):
        ring_kernel = np.ones((k, k), np.uint8)
        ring_kernel[1:-1, 1:-1] = 0
        inner_kernel = np.zeros((k, k), np.uint8)
        inner_kernel[1:-1, 1:-1] = 1

        min_ring = cv2.erode(result, ring_kernel)
        max_ring = cv2.dilate(result, ring_kernel)
        max_inner = cv2.dilate(result, inner_kernel)
        min_inner = cv2.erode(result, inner_kernel)

        result[max_inner < min_ring] = min_ring[max_inner < min_ring]
        result[min_inner > max_ring] = max_ring[min_inner > max_ring]
    return result


def apply_multipass_clahe(img, clip, grid, passes=1):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    if passes <= 1:
        return clahe.apply(img)
    h, w = img.shape
    acc = np.zeros((h, w), dtype=np.float32)
    for i in range(passes):
        shift = int((grid * i) / passes)
        padded = cv2.copyMakeBorder(img, shift, 0, shift, 0, cv2.BORDER_REFLECT)
        res = clahe.apply(padded)
        acc += res[shift : shift + h, shift : shift + w].astype(np.float32)
    return np.clip(acc / passes, 0, 255).astype(np.uint8)


def process_pipeline(img_uint8, cfg):
    current = img_uint8.copy()
    if cfg["use_fill"]:
        current = apply_progressive_hole_filling(current, cfg["fill_max_k"])
    if cfg["use_clahe"]:
        current = apply_multipass_clahe(
            current, cfg["clahe_clip"], cfg["clahe_grid"], cfg["clahe_passes"]
        )

    current_float = current.astype(np.float32)

    if cfg["use_bilat"]:
        current_float = cv2.bilateralFilter(
            current_float,
            d=cfg["bilat_d"],
            sigmaColor=cfg["bilat_sc"],
            sigmaSpace=cfg["bilat_ss"],
        )

    if cfg["use_nlm"]:
        temp = np.clip(current_float, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            temp, None, cfg["nlm_h"], cfg["nlm_t"], cfg["nlm_s"]
        )
        current_float = denoised.astype(np.float32)

    if cfg["use_lap"]:
        blur_k = cfg["lap_blur"]
        src = (
            cv2.GaussianBlur(current_float, (0, 0), sigmaX=blur_k)
            if blur_k > 0
            else current_float
        )
        lap = cv2.Laplacian(src, cv2.CV_32F, ksize=1)
        current_float = current_float - (lap * cfg["lap_str"])

    if cfg["use_ssaa"]:
        h, w = current_float.shape[:2]
        up_h, up_w = int(h * cfg["ssaa_factor"]), int(w * cfg["ssaa_factor"])
        upscaled = cv2.resize(
            current_float, (up_w, up_h), interpolation=cv2.INTER_CUBIC
        )
        if cfg["ssaa_smooth"] > 0:
            upscaled = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=cfg["ssaa_smooth"])
        current_float = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_AREA)

    return current_float


# --- MAIN ---


def main():
    if not os.path.exists(DIRS["input"]):
        print(
            f"❌ Ошибка: Папка {DIRS['input']} не существует. Сначала запустите скрипт нарезки (Step 1)."
        )
        return

    # 1. Рекурсивный поиск картинок
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    for ext in image_extensions:
        image_files.extend(
            glob(os.path.join(DIRS["input"], "**", "images", "**", ext), recursive=True)
        )

    if not image_files:
        print(f"⚠️ Изображения не найдены в подпапках {DIRS['input']}/images/...")
        return

    print(f"Найдено {len(image_files)} изображений для обработки.")

    for img_path in tqdm(image_files):
        # Формируем путь для сохранения
        rel_path = os.path.relpath(img_path, DIRS["input"])
        out_path = os.path.join(DIRS["output"], rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # A. Обработка изображения
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        processed_float = process_pipeline(img, CONFIG)

        # Финальная нормализация
        final_norm = cv2.normalize(processed_float, None, 0, 255, cv2.NORM_MINMAX)
        final_uint8 = np.clip(final_norm, 0, 255).astype(np.uint8)

        cv2.imwrite(out_path, final_uint8)

        # B. Копирование лейблов
        lbl_path_src = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        lbl_path_dst = out_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        if os.path.exists(lbl_path_src):
            os.makedirs(os.path.dirname(lbl_path_dst), exist_ok=True)
            shutil.copy2(lbl_path_src, lbl_path_dst)

    # C. ИСПРАВЛЕНИЕ DATASET.YAML
    yaml_src = os.path.join(DIRS["input"], "dataset.yaml")
    yaml_dst = os.path.join(DIRS["output"], "dataset.yaml")

    if os.path.exists(yaml_src):
        # 1. Читаем старый yaml
        with open(yaml_src, "r") as f:
            data = yaml.safe_load(f)

        # 2. Подменяем путь на АБСОЛЮТНЫЙ путь к новой папке
        abs_output_path = os.path.abspath(DIRS["output"])
        data["path"] = abs_output_path

        # 3. Сохраняем новый yaml
        with open(yaml_dst, "w") as f:
            yaml.dump(data, f, sort_keys=False)

        print(f"✅ Config обновлен: path -> {abs_output_path}")

    print(f"\n✅ Готово! Данные сохранены в {DIRS['output']}")


if __name__ == "__main__":
    main()
