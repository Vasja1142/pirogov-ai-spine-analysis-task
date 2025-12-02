"""
Streamlit-приложение для визуализации предсказаний модели YOLOv8 Segmentation
и сравнения их с истинными метками (Ground Truth).

Пользователь может:
- Загружать изображения из указанной директории.
- Переключаться между изображениями.
- Просматривать исходное изображение, истинные метки и предсказания модели.

Использование:
streamlit run scripts/08_predict_visualize_streamlit.py -- \
--model-path data/05_runs/spine_segmentation_v2/weights/best.pt \
--image-dir data/04_normalized/test/images
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Optional, Tuple

st.set_page_config(layout="wide", page_title="YOLOv8 Segmentation Visualizer")
st.title("🩻 YOLOv8 Segmentation Visualizer")
st.markdown("Сравнение предсказаний модели с истинными метками.")

# ============================================================================
# ⚙️ ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ И МОДЕЛИ
# ============================================================================

def load_image_paths(image_directory: Path) -> List[Path]:
    """Загружает список путей ко всем поддерживаемым изображениям в директории."""
    if not image_directory.is_dir():
        st.error(f"Ошибка: Директория с изображениями не найдена: {image_directory}")
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    image_paths = sorted([p for ext in extensions for p in image_directory.glob(ext)])
    if not image_paths:
        st.warning(f"В директории {image_directory} не найдено изображений.")
    return image_paths

def load_yolo_model(model_path: Path) -> Optional[YOLO]:
    """Загружает модель YOLOv8 из указанного пути."""
    if not model_path.is_file():
        st.error(f"Ошибка: Файл модели не найден: {model_path}")
        return None
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели YOLO: {e}")
        return None

# ============================================================================
# 🎨 ФУНКЦИИ ОТРИСОВКИ
# ============================================================================

def get_ground_truth_and_draw(img_path: Path, original_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Загружает метки для изображения и отрисовывает их.

    Args:
        img_path: Путь к файлу изображения.
        original_image: Исходное изображение NumPy array.

    Returns:
        Изображение с отрисованными метками или None, если меток нет.
    """
    # Предполагаем, что labels находится на два уровня выше images (в ../../labels/split/)
    # Например: data/04_normalized/test/images/img.jpg -> data/04_normalized/test/labels/img.txt
    label_path = img_path.parents[1] / "labels" / f"{img_path.stem}.txt"

    if not label_path.is_file():
        return None # Нет файла разметки

    gt_image = original_image.copy()
    # Конвертируем в RGB, если нужно, для отрисовки цветных полигонов
    if gt_image.ndim == 2 or gt_image.shape[2] == 1: # Grayscale
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_GRAY2RGB)
    else: # BGR (OpenCV default)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

    h, w = gt_image.shape[:2]
    has_labels = False

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue

                # Предполагаем формат YOLO: class_id x_c y_c w h (для bbox) или class_id x1 y1 x2 y2 ... (для poly)
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                points = coords.astype(np.int32)

                # Отрисовка полигона
                cv2.polylines(gt_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                has_labels = True
    except Exception as e:
        st.warning(f"Ошибка чтения или отрисовки файла разметки {label_path.name}: {e}")
        return None

    return gt_image if has_labels else None

def display_images(
    original_rgb: np.ndarray,
    gt_rgb: Optional[np.ndarray],
    pred_rgb: np.ndarray,
    img_name: str,
    current_index: int,
    total_images: int
):
    """
    Отображает три изображения (оригинал, GT, предсказание) в Streamlit.
    """
    st.subheader(f"Файл: {img_name} [{current_index + 1}/{total_images}]")

    cols = st.columns(3)

    with cols[0]:
        st.image(original_rgb, caption="Оригинал", use_column_width=True)
    with cols[1]:
        if gt_rgb is not None:
            st.image(gt_rgb, caption="Разметка (Manual GT)", use_column_width=True)
        else:
            st.markdown("<div style='text-align: center; color: gray;'>Нет файла разметки</div>", unsafe_allow_html=True)
            st.caption("Разметка отсутствует")
    with cols[2]:
        st.image(pred_rgb, caption="Результат Модели", use_column_width=True)

# ============================================================================
# 🚀 ОСНОВНАЯ ЛОГИКА STREAMLIT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Visualizer.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/05_runs/spine_segmentation_v2/weights/best.pt"),
        help="Путь к файлу обученной модели YOLO (например, best.pt)."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/04_normalized/test/images"),
        help="Путь к директории с изображениями для визуализации."
    )
    # Необходимо для корректной работы streamlit run с аргументами
    # https://docs.streamlit.io/library/advanced-features/command-line-options
    args = parser.parse_args()

    # Инициализация состояния Streamlit
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = load_image_paths(args.image_dir)
        st.session_state.current_index = 0
    if "model" not in st.session_state:
        st.session_state.model = load_yolo_model(args.model_path)

    image_paths = st.session_state.image_paths
    model = st.session_state.model

    if not image_paths or model is None:
        st.stop()

    total_images = len(image_paths)
    current_index = st.session_state.current_index
    img_path = image_paths[current_index]

    # --- Боковая панель для навигации ---
    st.sidebar.header("Навигация")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("⬅️ Предыдущее", key="prev_img"):
            st.session_state.current_index = (current_index - 1 + total_images) % total_images
            st.experimental_rerun()
    with col2:
        if st.button("Следующее ➡️", key="next_img"):
            st.session_state.current_index = (current_index + 1) % total_images
            st.experimental_rerun()

    st.sidebar.write(f"Текущее изображение: {current_index + 1} из {total_images}")

    # --- Основная область отображения ---
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        st.error(f"Не удалось загрузить изображение: {img_path.name}")
        return
    
    # Конвертация для отображения (Streamlit ожидает RGB)
    if original_img.ndim == 2 or original_img.shape[2] == 1:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Получение предсказаний модели
    results = model(original_img, retina_masks=True, verbose=False, conf=0.25)
    pred_plot = results[0].plot(boxes=False, conf=True)
    pred_rgb = cv2.cvtColor(pred_plot, cv2.COLOR_BGR2RGB)

    # Получение и отрисовка Ground Truth
    gt_rgb = get_ground_truth_and_draw(img_path, original_img)

    display_images(
        original_rgb, gt_rgb, pred_rgb, img_path.name, current_index, total_images
    )

if __name__ == "__main__":
    main()
