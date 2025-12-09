import os
import numpy as np
import cv2
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# --- ПОПЫТКА ИМПОРТА CXAS ---
try:
    from cxas.label_mapper import id2label_dict
except ImportError:
    print("❌ Ошибка: Библиотека cxas не найдена.")
    exit()

# --- НАСТРОЙКИ ---
RAW_IMG_DIR = "data/01_raw/PAX-RayPlusPlus/images_patlas"
RAW_LBL_DIR = "data/01_raw/PAX-RayPlusPlus/labels"
OUTPUT_DIR = "data/02_processed"

# Отступ вокруг позвоночника (в процентах от размера найденной зоны)
PADDING_PCT = 0.05

# Целевые классы: T1 ... T12
TARGET_NAMES = [f"vertebrae T{i}" for i in range(1, 13)]


def get_roi_bounding_box(masks_dict, height, width):
    """
    Находит КВАДРАТНУЮ зону интереса вокруг T1-T12.
    """
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    found_any = False

    # 1. Находим границы самих позвонков
    for name, mask in masks_dict.items():
        if mask.max() > 0:
            found_any = True
            y_indices, x_indices = np.where(mask > 0)
            min_x = min(min_x, x_indices.min())
            max_x = max(max_x, x_indices.max())
            min_y = min(min_y, y_indices.min())
            max_y = max(max_y, y_indices.max())

    if not found_any:
        return None

    # Высота и ширина позвоночного столба
    spine_h = max_y - min_y
    spine_w = max_x - min_x

    # Центр позвоночника по X и по Y
    center_x = min_x + spine_w / 2
    center_y = min_y + spine_h / 2

    # 2. Определяем размер квадрата
    # Берем высоту позвоночника + отступ (например, 10% сверху и 10% снизу)
    # И делаем это стороной квадрата
    target_size = int(spine_h * (1 + PADDING_PCT * 2))

    # Половина стороны квадрата
    half_size = target_size / 2

    # 3. Вычисляем координаты квадрата от центра
    crop_x1 = int(center_x - half_size)
    crop_x2 = int(center_x + half_size)

    crop_y1 = int(center_y - half_size)
    crop_y2 = int(center_y + half_size)

    # 4. Проверка границ (чтобы не вылезти за пределы картинки)
    # Если квадрат вылезает влево - сдвигаем вправо
    if crop_x1 < 0:
        crop_x2 += abs(crop_x1)
        crop_x1 = 0
    # Если вылезает вправо
    if crop_x2 > width:
        crop_x1 -= crop_x2 - width
        crop_x2 = width

    # То же для высоты (хотя T1-T12 обычно в центре)
    if crop_y1 < 0:
        crop_y2 += abs(crop_y1)
        crop_y1 = 0
    if crop_y2 > height:
        crop_y1 -= crop_y2 - height
        crop_y2 = height

    # Финальная защита (если картинка узкая и квадрат физически не влезает)
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(width, crop_x2)
    crop_y2 = min(height, crop_y2)

    return (crop_x1, crop_y1, crop_x2, crop_y2)


def mask_to_yolo_polygon(mask, crop_coords):
    """
    Превращает маску в полигон с учетом смещения (кропа).
    crop_coords: (x1, y1, x2, y2) - координаты выреза
    """
    c_x1, c_y1, c_x2, c_y2 = crop_coords
    crop_w = c_x2 - c_x1
    crop_h = c_y2 - c_y1

    # Находим контуры на исходной маске
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Фильтр шума
            poly = cnt.reshape(-1, 2).astype(np.float32)

            # --- СМЕЩЕНИЕ И НОРМАЛИЗАЦИЯ ---
            # 1. Вычитаем координаты начала кропа
            poly[:, 0] -= c_x1
            poly[:, 1] -= c_y1

            # 2. Нормализуем на новый размер
            poly[:, 0] /= crop_w
            poly[:, 1] /= crop_h

            # 3. Обрезаем значения, если вдруг вылезли за 0..1 (из-за сглаживания)
            poly = np.clip(poly, 0.0, 1.0)

            polygons.append(poly.flatten().tolist())

    return polygons


def main():
    # 1. Собираем маппинг ID из cxas
    # Нам нужно знать оригинальные ID для T1...T12
    # И создать новые ID для YOLO (0...11)

    # { "vertebrae T1": original_id_15, ... }
    name_to_orig_id = {}
    for idx_str, name in id2label_dict.items():
        if name in TARGET_NAMES:
            name_to_orig_id[name] = int(idx_str)

    # Проверка, все ли классы найдены в библиотеке
    if len(name_to_orig_id) != 12:
        print(
            f"⚠️ Внимание: Найдено только {len(name_to_orig_id)} из 12 классов T1-T12 в библиотеке cxas."
        )

    # Маппинг для нового датасета: "vertebrae T1" -> 0, "vertebrae T2" -> 1 ...
    # Мы сортируем TARGET_NAMES, чтобы T1 был 0, T10 был 9 и т.д.
    # Сортировка строк "T1", "T10" специфична, лучше задать жестко
    sorted_targets = [f"vertebrae T{i}" for i in range(1, 13)]
    new_class_map = {name: i for i, name in enumerate(sorted_targets)}

    # 2. Подготовка папок
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # 3. Фильтрация файлов (Lateral Only)
    all_files = glob(os.path.join(RAW_IMG_DIR, "*.png"))
    # Ищем файлы, в имени которых есть 'lateral'
    lateral_files = [f for f in all_files if "lateral" in os.path.basename(f).lower()]

    if not lateral_files:
        print("❌ Не найдено снимков с пометкой 'lateral'!")
        exit()

    print(f"🔍 Найдено боковых снимков: {len(lateral_files)} (из {len(all_files)})")

    train_files, val_files = train_test_split(
        lateral_files, test_size=0.2, random_state=42
    )

    # 4. Обработка
    processed_count = 0
    skipped_count = 0

    for split, files in zip(["train", "val"], [train_files, val_files]):
        print(f"\n🚀 Обработка {split}...")

        for img_path in tqdm(files):
            basename = os.path.basename(img_path)
            npz_path = os.path.join(RAW_LBL_DIR, basename.replace(".png", ".npz"))

            if not os.path.exists(npz_path):
                continue

            try:
                # Загружаем маски
                data = np.load(npz_path)
                # Ищем ключ с данными
                key = "data" if "data" in data else list(data.keys())[0]
                full_mask = data[key]
                # Fix shape (159, H, W)
                if full_mask.shape[0] != 159 and full_mask.shape[-1] == 159:
                    full_mask = np.moveaxis(full_mask, -1, 0)

                _, h, w = full_mask.shape

                # --- СТРОГАЯ ПРОВЕРКА (ВСЕ ОТ T1 ДО T12) ---
                missing_vertebrae = False

                # Проходим по списку ["vertebrae T1", "vertebrae T2", ... "T12"]
                for t_name in sorted_targets:
                    orig_id = name_to_orig_id.get(t_name)

                    # Если такого класса вообще нет в библиотеке или маска пустая
                    if orig_id is None or full_mask[orig_id].max() == 0:
                        missing_vertebrae = True
                        break  # Дальше можно не проверять, снимок бракованный

                if missing_vertebrae:
                    skipped_count += 1
                    continue  # Пропускаем этот файл, не сохраняем!

                # Если код дошел сюда, значит ВСЕ T1-T12 на месте.

                # --- СБОР ВСЕХ НУЖНЫХ МАСОК ---
                current_spine_masks = {}
                for name in sorted_targets:
                    # Мы уже проверили выше, что они есть, можно смело брать
                    orig_id = name_to_orig_id.get(name)
                    current_spine_masks[name] = full_mask[orig_id]

                # --- ВЫЧИСЛЕНИЕ КРОПА ---
                crop_box = get_roi_bounding_box(current_spine_masks, h, w)
                if crop_box is None:
                    skipped_count += 1
                    continue

                c_x1, c_y1, c_x2, c_y2 = crop_box

                # --- ОБРЕЗКА И СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ ---
                # Читаем оригинал
                img_cv = cv2.imread(img_path)  # BGR
                # Вырезаем
                img_cropped = img_cv[c_y1:c_y2, c_x1:c_x2]

                # Сохраняем
                out_img_path = os.path.join(OUTPUT_DIR, "images", split, basename)
                cv2.imwrite(out_img_path, img_cropped)

                # --- ГЕНЕРАЦИЯ ЛЕЙБЛОВ ---
                txt_lines = []
                for name, mask in current_spine_masks.items():
                    # Если позвонок есть на снимке
                    if mask.max() > 0:
                        polygons = mask_to_yolo_polygon(mask, crop_box)
                        class_id = new_class_map[name]  # 0..11

                        for poly in polygons:
                            line = f"{class_id} " + " ".join(map(str, poly))
                            txt_lines.append(line)

                # Сохраняем txt
                out_txt_path = os.path.join(
                    OUTPUT_DIR, "labels", split, basename.replace(".png", ".txt")
                )
                with open(out_txt_path, "w") as f:
                    f.write("\n".join(txt_lines))

                processed_count += 1

            except Exception as e:
                print(f"Ошибка с файлом {basename}: {e}")
                continue

    # 5. СОЗДАЕМ CONFIG
    # Словарь: {0: 'vertebrae T1', 1: 'vertebrae T2'...}
    yaml_names = {i: name for name, i in new_class_map.items()}

    yaml_data = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": yaml_names,
    }

    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print("\n✅ Обработка завершена!")
    print(f"Сохранено снимков: {processed_count}")
    print(f"Отброшено (нет T1/T12 или не боковые): {skipped_count}")
    print(f"Результат в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
