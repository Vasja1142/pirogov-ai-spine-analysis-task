import os
import numpy as np
import cv2
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# --- ПОПЫТКА ИМПОРТА КАРТЫ КЛАССОВ ИЗ CXAS ---
try:
    from cxas.label_mapper import id2label_dict

    print("✅ Библиотека CXAS найдена. Карта классов загружена.")
except ImportError:
    print("❌ ОШИБКА: Библиотека cxas не найдена или установлена неправильно!")
    print("Убедитесь, что вы запускаете скрипт через 'poetry run python ...'")
    exit()

# --- НАСТРОЙКИ ---
# Пути к вашим данным
raw_images_dir = "data/01_raw/PAX-RayPlusPlus/images_patlas"
raw_labels_dir = "data/01_raw/PAX-RayPlusPlus/labels"
output_dir = "data/02_yolo_spine"


def mask_to_polygon(mask):
    """Превращает бинарную маску в полигоны YOLO (x y x y ...)"""
    h, w = mask.shape
    # Находим контуры
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        # Фильтруем совсем мелкий шум
        if cv2.contourArea(cnt) > 20:
            poly = cnt.reshape(-1, 2).astype(np.float32)
            # Нормализация координат (0..1)
            poly[:, 0] /= w
            poly[:, 1] /= h
            # Проверка на валидность координат
            if (poly >= 0).all() and (poly <= 1).all():
                polygons.append(poly.flatten().tolist())
    return polygons


def main():
    # 1. СОЗДАЕМ КАРТУ ПОЗВОНКОВ
    # Нам нужно узнать, под какими индексами (0..158) лежат позвонки

    # Словарь: {оригинальный_индекс_в_npz: "имя_класса"}
    spine_indices = {}

    print("\n🔍 Фильтруем классы (ищем 'vertebrae')...")
    for idx_str, name in id2label_dict.items():
        # idx_str может быть строкой или числом, приводим к int
        idx = int(idx_str)
        if "vertebrae" in name.lower() or "spine" in name.lower():
            spine_indices[idx] = name

    if not spine_indices:
        print("❌ Не найдено классов позвоночника в cxas!")
        exit()

    # Сортируем по оригинальному индексу
    sorted_orig_indices = sorted(spine_indices.keys())

    # Создаем маппинг для YOLO: 0 -> первый позвонок, 1 -> второй...
    # original_idx -> yolo_id
    orig_to_yolo = {orig: i for i, orig in enumerate(sorted_orig_indices)}

    # yolo_id -> "имя" (для yaml файла)
    yolo_names = {i: spine_indices[orig] for i, orig in enumerate(sorted_orig_indices)}

    print(f"✅ Выбрано {len(yolo_names)} классов позвонков.")
    print(f"Пример: ID {sorted_orig_indices[0]} -> YOLO 0 ({yolo_names[0]})")

    # 2. ПОДГОТОВКА ПАПОК
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # 3. СПИСОК ФАЙЛОВ
    image_files = sorted(glob(os.path.join(raw_images_dir, "*.png")))
    if not image_files:
        print(f"❌ Картинки не найдены в {raw_images_dir}")
        exit()

    train_files, val_files = train_test_split(
        image_files, test_size=0.2, random_state=42
    )

    # 4. ОБРАБОТКА ФАЙЛОВ
    for split, files in zip(["train", "val"], [train_files, val_files]):
        print(f"\n🚀 Обработка {split} ({len(files)} фото)...")

        for img_path in tqdm(files):
            basename = os.path.basename(img_path)
            npz_name = basename.replace(".png", ".npz")
            npz_path = os.path.join(raw_labels_dir, npz_name)

            if not os.path.exists(npz_path):
                continue

            try:
                # Загружаем массив. Ключ 'data' мы узнали из диагностики
                # shape = (159, H, W)
                full_mask = np.load(npz_path)["data"]

                # Если вдруг маска (H, W, C), транспонируем
                if full_mask.shape[0] != 159 and full_mask.shape[-1] == 159:
                    full_mask = np.moveaxis(full_mask, -1, 0)

            except Exception as e:
                print(f"Ошибка чтения {npz_name}: {e}")
                continue

            txt_lines = []

            # Бежим только по тем слоям, которые являются позвонками
            for orig_idx, yolo_id in orig_to_yolo.items():
                if orig_idx >= full_mask.shape[0]:
                    continue

                # Достаем слой конкретного позвонка
                mask_layer = full_mask[orig_idx]

                # В файле boolean (True/False), переводим в uint8
                if mask_layer.max():  # Если там есть хоть что-то (не все False)
                    polygons = mask_to_polygon(mask_layer)

                    for poly in polygons:
                        # Формат: <class_id> <x1> <y1> <x2> <y2> ...
                        line = f"{yolo_id} " + " ".join(map(str, poly))
                        txt_lines.append(line)

            # Если нашли позвонки на этом снимке
            if txt_lines:
                # 1. Копируем картинку
                shutil.copy(
                    img_path, os.path.join(output_dir, "images", split, basename)
                )

                # 2. Сохраняем лейблы
                txt_name = basename.replace(".png", ".txt")
                with open(
                    os.path.join(output_dir, "labels", split, txt_name), "w"
                ) as f:
                    f.write("\n".join(txt_lines))

    # 5. СОЗДАЕМ DATASET.YAML
    yaml_data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": yolo_names,
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n🎉 Датасет готов! Конфиг сохранен: {yaml_path}")


if __name__ == "__main__":
    main()
