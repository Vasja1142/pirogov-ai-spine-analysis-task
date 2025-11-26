import streamlit as st
import cv2
import numpy as np
import albumentations as A
from PIL import Image

st.set_page_config(layout="wide", page_title="X-Ray Tuner v3 (Big Zoom)")

st.title("🩻 X-Ray Tuner v3: Big Vision")
st.markdown("Теперь с автоматическим увеличением маленьких снимков.")

# --- ФУНКЦИЯ ДЛЯ УМНОГО РЕСАЙЗА ---
def smart_resize(img, target_width=1024):
    h, w = img.shape[:2]
    # Если картинка и так большая, не трогаем её, если не просят
    if w < target_width:
        scale = target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # INTER_CUBIC — лучшее качество для увеличения рентгена (мягкие края)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("🔍 Масштабирование")
target_size = st.sidebar.slider("Целевая ширина просмотра (px)", 512, 2048, 1280, step=128)

st.sidebar.divider()
st.sidebar.header("1. Шум и Детали")

# Фильтры
use_bilateral = st.sidebar.checkbox("Bilateral (Сгладить, но оставить края)", value=False)
bil_d = st.sidebar.slider("Diameter", 1, 20, 9)
bil_sigmaColor = st.sidebar.slider("Sigma Color", 10, 150, 75)
bil_sigmaSpace = st.sidebar.slider("Sigma Space", 10, 150, 75)

use_median = st.sidebar.checkbox("Median Blur (Убрать зерно)", value=False)
median_k = st.sidebar.slider("Kernel Size", 3, 11, 3, step=2)

st.sidebar.header("2. Контраст")
use_clahe = st.sidebar.checkbox("CLAHE (Гистограмма)", value=True)
clahe_limit = st.sidebar.slider("Clip Limit", 1.0, 20.0, 4.0, 0.1)
clahe_grid = st.sidebar.slider("Grid Size", 2, 64, 8)

use_gamma = st.sidebar.checkbox("Gamma (Яркость)", value=False)
gamma_value = st.sidebar.slider("Gamma Value", 50, 400, 100)

use_sharpen = st.sidebar.checkbox("Sharpen (Резкость)", value=False)
sharpen_alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5)
sharpen_light = st.sidebar.slider("Lightness", 0.5, 2.0, 1.0)

use_invert = st.sidebar.checkbox("Invert (Негатив)", value=False)

# --- ЗАГРУЗКА ---
uploaded_file = st.file_uploader("Загрузи снимок", type=["jpg", "png", "jpeg", "bmp", "tif"])

if uploaded_file is not None:
    # Чтение
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_raw = cv2.imdecode(file_bytes, 1)

    # Конвертация цвета
    if len(original_raw.shape) == 2:
        original_raw = cv2.cvtColor(original_raw, cv2.COLOR_GRAY2RGB)
    else:
        original_raw = cv2.cvtColor(original_raw, cv2.COLOR_BGR2RGB)

    # --- ПРИМЕНЕНИЕ ФИЛЬТРОВ ---
    # Важно: Сначала применяем фильтры на ОРИГИНАЛЬНОМ размере (чтобы не исказить данные),
    # а увеличиваем только для показа на экране.
    
    processed_image = original_raw.copy()
    
    # 0. OpenCV Pre-processing
    if use_bilateral:
        processed_image = cv2.bilateralFilter(processed_image, bil_d, bil_sigmaColor, bil_sigmaSpace)
    
    # 1. Albumentations
    transforms_list = []
    if use_median: transforms_list.append(A.MedianBlur(blur_limit=(median_k, median_k), p=1.0))
    if use_clahe: transforms_list.append(A.CLAHE(clip_limit=(clahe_limit, clahe_limit), tile_grid_size=(clahe_grid, clahe_grid), p=1.0))
    if use_gamma: transforms_list.append(A.RandomGamma(gamma_limit=(gamma_value, gamma_value), p=1.0))
    if use_sharpen: transforms_list.append(A.Sharpen(alpha=(sharpen_alpha, sharpen_alpha), lightness=(sharpen_light, sharpen_light), p=1.0))
    if use_invert: transforms_list.append(A.InvertImg(p=1.0))

    if transforms_list:
        transform = A.Compose(transforms_list)
        final_result = transform(image=processed_image)['image']
    else:
        final_result = processed_image

    # --- ВИЗУАЛИЗАЦИЯ (УВЕЛИЧЕНИЕ) ---
    # Теперь готовим картинки для вывода на экран, растягивая их
    view_original = smart_resize(original_raw, target_size)
    view_result = smart_resize(final_result, target_size)

    try:
        from streamlit_image_comparison import image_comparison
        st.subheader("Сравнение")
        # Этот компонент сам умеет растягиваться, но мы подаем ему уже увеличенные картинки
        image_comparison(
            img1=view_original,
            img2=view_result,
            label1="Оригинал",
            label2="Обработка",
            width=target_size, # Используем ширину из слайдера
            starting_position=50,
            show_labels=True,
            make_responsive=True, # Адаптивность
            in_memory=True
        )
    except ImportError:
        st.warning("Нет библиотеки сравнения, показываю рядом.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(view_original, caption="Оригинал", use_container_width=True)
        with col2:
            st.image(view_result, caption="Результат", use_container_width=True)

    # Инфо о размере
    h, w = original_raw.shape[:2]
    st.caption(f"Исходный размер: {w}x{h} px. | Отображается как: {view_original.shape[1]}x{view_original.shape[0]} px.")