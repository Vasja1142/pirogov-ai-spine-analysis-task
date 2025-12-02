"""
Streamlit-приложение для интерактивной настройки параметров обработки изображений и аугментации.

Позволяет пользователю загружать изображения (опционально с файлами разметки YOLO),
применять различные методы предобработки и аугментации, а также визуально
оценивать их влияние на изображение и, при необходимости, на полигональную разметку.

Основные функции:
- Загрузка изображений и файлов меток (.txt).
- Различные методы нормализации (Robust Auto-Levels, CLAHE, Bilateral, Median).
- Геометрические аугментации (S-Curve).
- Текстурные аугментации (Albumentations: ElasticTransform, ISONoise, Blur).
- Наложение облачного шума.
- Визуализация преобразований с помощью `streamlit_image_comparison`.
- Отображение полигонов на аугментированных изображениях.
"""

import streamlit as st
import cv2
import numpy as np
import albumentations as A
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Импортируем нашу новую библиотеку
from lib.image_processing import (
    smart_resize,
    robust_auto_levels,
    unsharp_mask_cv,
    apply_fast_cloud_noise,
    apply_advanced_spine_curve,
    apply_albumentations,
)

# Для сравнения изображений
try:
    from streamlit_image_comparison import image_comparison
except ImportError:
    st.warning("Для полноценного сравнения установите: pip install streamlit-image-comparison")
    image_comparison = None

st.set_page_config(layout="wide", page_title="X-Ray Tuner Unified")
st.title("🩻 X-Ray Tuner: Unified")
st.markdown("Объединенная и улучшенная версия тюнера с поддержкой полигонов.")

# ============================================================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def load_polygons_from_txt(label_file_content: str) -> List[List[float]]:
    """
    Загружает полигоны из содержимого .txt файла в формате YOLO.
    Возвращает список списков [class_id, x1, y1, x2, y2, ...].
    """
    polygons = []
    for line in label_file_content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:  # class_id + min 2 points (x,y,x,y)
            try:
                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                if len(coords) % 2 == 0:  # Координаты должны быть парами
                    polygons.append([class_id] + coords)
            except ValueError:
                st.warning(f"Пропущена некорректная строка в файле меток: {line}")
    return polygons

def draw_polygons_on_image(image: np.ndarray, polygons: List[list], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Отрисовывает полигоны на изображении.

    Args:
        image: Изображение, на котором нужно отрисовать полигоны (OpenCV BGR или RGB).
        polygons: Список полигонов в формате [class_id, x1, y1, x2, y2, ...],
                  где координаты нормализованы (0.0-1.0).
        color: Цвет полигонов (BGR формат).

    Returns:
        Изображение с отрисованными полигонами.
    """
    if not polygons or image is None: # Защита от пустого списка или пустого изображения
        return image

    display_image = image.copy()
    h, w = display_image.shape[:2]

    for poly in polygons:
        # Пропускаем class_id и берем только координаты
        coords_flat = poly[1:]
        # Преобразуем плоский список в массив NumPy (N, 2)
        points = np.array(coords_flat).reshape(-1, 2)

        # Денормализуем координаты (из 0-1 в пиксели)
        points[:, 0] *= w
        points[:, 1] *= h

        # Округляем до целых чисел и приводим к типу int32 для cv2.polylines
        points = points.astype(np.int32)

        # Отрисовываем полигон. isClosed=True замыкает фигуру.
        cv2.polylines(display_image, [points], isClosed=True, color=color, thickness=2)
        
        # Опционально: отрисовка class_id у первой точки
        if len(points) > 0:
            class_id = poly[0]
            cv2.putText(
                display_image, 
                f"ID:{class_id}", 
                tuple(points[0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1,
                cv2.LINE_AA
            )
    return display_image

# ============================================================================
# 🎨 ИНТЕРФЕЙС STREAMLIT
# ============================================================================

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("🔍 Просмотр")
target_size = st.sidebar.slider("Размер превью (px)", 512, 2048, 1024, step=128)

st.sidebar.divider()

st.sidebar.header("1. Предобработка")
norm_method = st.sidebar.radio("Метод нормализации", ["Нет", "Robust Auto-Levels", "CLAHE"])

if norm_method == "Robust Auto-Levels":
    st.sidebar.caption("Параметры 'Умной' нормализации")
    robust_cutoff = st.sidebar.slider("Cutoff Percent", 0.0, 1.0, 0.5, 0.01)
    robust_sharpen = st.sidebar.slider("Sharpen Amount", 0.0, 5.0, 1.5, 0.1)
    robust_sigma = st.sidebar.slider("Sharpen Radius (Sigma)", 0.0, 50.0, 10.0, 1.0)
elif norm_method == "CLAHE":
    st.sidebar.caption("Параметры CLAHE")
    clahe_limit = st.sidebar.slider("Clip Limit", 1.0, 20.0, 4.0, 0.1)
    clahe_grid = st.sidebar.slider("Grid Size", 2, 64, 8)

st.sidebar.subheader("Доп. фильтры")
use_bilateral = st.sidebar.checkbox("Bilateral Filter", value=False)
if use_bilateral:
    bil_d = st.sidebar.slider("Diameter", 1, 20, 9)
    bil_sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
    bil_sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)

use_median = st.sidebar.checkbox("Median Blur", value=False)
if use_median:
    median_k = st.sidebar.slider("Kernel Size", 3, 11, 3, step=2)

st.sidebar.divider()

st.sidebar.header("2. Аугментация")
use_augmentation = st.sidebar.checkbox("Включить аугментацию", value=False)
process_polygons = st.sidebar.checkbox("Обрабатывать и визуализировать полигоны", value=False)

if use_augmentation:
    aug_count = st.sidebar.slider("Количество примеров", 1, 6, 3)

    st.sidebar.subheader("🦴 Геометрия (S-Curve)")
    use_spine_curve = st.sidebar.checkbox("Изгиб (S-Curve)", value=True)
    if use_spine_curve:
        amp_val = st.sidebar.slider("Амплитуда", 0.01, 0.20, 0.10, 0.01)
        per_val = st.sidebar.slider("Частота (Periods)", 0.1, 2.0, 1.0, 0.1)
        pha_val = st.sidebar.slider("Фаза", 0.0, 6.28, 0.0, 0.1)

    st.sidebar.subheader("🎨 Текстура и Шум")
    use_cloud = st.sidebar.checkbox("Облачный шум (Cloud Noise)", value=True)
    if use_cloud:
        cloud_intensity = st.sidebar.slider("Cloud Intensity", 0.1, 1.0, 0.6, 0.1)
        cloud_blur = st.sidebar.slider("Cloud Scale", 0.1, 2.0, 1.0, 0.1)

    use_albu = st.sidebar.checkbox("Albumentations (Elastic/ISO/Blur)", value=True)
    if use_albu:
        ela_alpha = st.sidebar.slider("Elastic Alpha", 50, 200, 120, 10)
        ela_sigma = st.sidebar.slider("Elastic Sigma", 1.0, 20.0, 6.0, 0.5)
        ela_affine = st.sidebar.slider("Elastic Affine", 1.0, 10.0, 3.6, 0.1)
        iso_int_min = st.sidebar.slider("ISO Intensity Min", 0.0, 1.0, 0.1, 0.05)
        iso_int_max = st.sidebar.slider("ISO Intensity Max", 0.0, 1.0, 0.5, 0.05)
        blur_prob = st.sidebar.slider("Blur Probability", 0.0, 1.0, 0.2, 0.1)

# --- ЗАГРУЗКА ФАЙЛОВ ---
uploaded_img_file = st.file_uploader(
    "Загрузи снимок", type=["jpg", "png", "jpeg", "bmp", "tif"]
)
uploaded_label_file = None
if process_polygons:
    uploaded_label_file = st.file_uploader("Загрузи .txt файл разметки (YOLO формат)", type=["txt"])

if uploaded_img_file is not None:
    # --- Чтение и подготовка изображения ---
    file_bytes = np.asarray(bytearray(uploaded_img_file.read()), dtype=np.uint8)
    original_raw = cv2.imdecode(file_bytes, 1)
    if original_raw is None:
        st.error("Не удалось загрузить изображение. Проверьте формат файла.")
        st.stop()

    # Конвертация в RGB для унификации обработки (если не RGB)
    if original_raw.ndim == 2 or original_raw.shape[2] == 1: # Grayscale
        original_rgb = cv2.cvtColor(original_raw, cv2.COLOR_GRAY2RGB)
    else: # BGR (OpenCV default) to RGB
        original_rgb = cv2.cvtColor(original_raw, cv2.COLOR_BGR2RGB)

    # --- Загрузка и обработка полигонов ---
    polygons: List[List[float]] = []
    if process_polygons and uploaded_label_file is not None:
        label_content = uploaded_label_file.read().decode("utf-8")
        polygons = load_polygons_from_txt(label_content)
        if not polygons:
            st.warning("Файл разметки загружен, но полигоны не найдены или некорректны.")

    # --- 1. ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ ---
    processed_image = original_rgb.copy()

    if use_bilateral:
        processed_image = cv2.bilateralFilter(processed_image, bil_d, bil_sigmaColor, bil_sigmaSpace)
    if use_median:
         processed_image = cv2.medianBlur(processed_image, median_k)

    if norm_method == "Robust Auto-Levels":
        # Обработка L-канала в LAB пространстве
        lab = cv2.cvtColor(processed_image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = robust_auto_levels(l_channel, robust_cutoff)
        l_channel = unsharp_mask_cv(l_channel, robust_sharpen, robust_sigma)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif norm_method == "CLAHE":
        # Albumentations CLAHE работает с RGB напрямую
        transform = A.CLAHE(clip_limit=clahe_limit, tile_grid_size=(clahe_grid, clahe_grid), p=1.0)
        processed_image = transform(image=processed_image)["image"]

    final_result = processed_image
    final_polygons_after_preprocessing = polygons.copy() # Полигоны не меняются при предобработке

    # --- 2. ВИЗУАЛИЗАЦИЯ ПРЕДОБРАБОТКИ ---
    if image_comparison:
        st.subheader("Сравнение предобработки")
        view_original = smart_resize(original_rgb, target_size)
        view_result = smart_resize(final_result, target_size)

        if process_polygons and polygons:
            # Отрисовываем полигоны и на оригинале, и на обработанном
            view_original = draw_polygons_on_image(view_original, polygons, (255, 0, 0)) # Синий для оригинала
            view_result = draw_polygons_on_image(view_result, final_polygons_after_preprocessing, (0, 0, 255)) # Красный для обработанного

        image_comparison(
            img1=view_original,
            img2=view_result,
            label1="Оригинал",
            label2="После обработки",
            width=target_size
        )
    else:
        st.subheader("Оригинал и обработанное изображение")
        col1, col2 = st.columns(2)
        with col1:
            st.image(smart_resize(original_rgb, target_size), caption="Оригинал", use_column_width=True)
        with col2:
            st.image(smart_resize(final_result, target_size), caption="После обработки", use_column_width=True)

    # --- 3. АУГМЕНТАЦИЯ ---
    if use_augmentation:
        st.divider()
        st.subheader("🧪 Результаты Аугментации")

        # Убедимся, что количество колонок не превышает aug_count
        cols = st.columns(min(aug_count, 3)) # Максимум 3 колонки для лучшей читаемости

        for i in range(aug_count):
            aug_img = final_result.copy()
            aug_polygons = [p.copy() for p in final_polygons_after_preprocessing] if process_polygons else []

            # 1. Изгиб
            if use_spine_curve:
                aug_img, aug_polygons = apply_advanced_spine_curve(
                    aug_img,
                    aug_polygons,
                    amp_val,
                    per_val,
                    pha_val,
                    random.choice([-1, 1]),
                )

            # 2. Текстуры (Albumentations)
            if use_albu:
                aug_img, aug_polygons = apply_albumentations(
                    img=aug_img,
                    polygons=aug_polygons,
                    elastic_alpha=ela_alpha,
                    elastic_sigma=ela_sigma,
                    elastic_affine=ela_affine,
                    iso_color=(0.01, 0.05),
                    iso_intensity=(iso_int_min, iso_int_max),
                    motion_p=blur_prob,
                    gauss_p=blur_prob,
                )

            # 3. Облачный шум
            if use_cloud:
                aug_img = apply_fast_cloud_noise(aug_img, cloud_intensity, cloud_blur)
            
            # 4. Рандомная инверсия
            if random.random() < 0.2: # Небольшой шанс инверсии
                aug_img = cv2.bitwise_not(aug_img)

            # Визуализация полигонов на аугментированном изображении
            if process_polygons and aug_polygons:
                aug_img_with_polys = draw_polygons_on_image(aug_img, aug_polygons, (0, 255, 255)) # Желтый для аугментаций
            else:
                aug_img_with_polys = aug_img

            with cols[i % len(cols)]:
                st.image(
                    smart_resize(aug_img_with_polys, target_size),
                    caption=f"Aug #{i + 1}",
                    use_container_width=True,
                )