import streamlit as st
import os
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from glob import glob
import random

# --- НАСТРОЙКИ ---
BASE_DIR = "data/02_processed"
YAML_PATH = os.path.join(BASE_DIR, "dataset.yaml")

# Настройка страницы
st.set_page_config(layout="wide", page_title="YOLO Dataset Inspector")

# --- ФУНКЦИИ ---


def load_class_names():
    """Загружает названия классов из dataset.yaml"""
    if not os.path.exists(YAML_PATH):
        st.error(f"Не найден конфиг {YAML_PATH}")
        return {}

    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
        return data.get("names", {})


def get_color(class_id):
    """Генерирует уникальный цвет для класса (яркий)"""
    # Используем colormap из matplotlib для генерации RGB
    cmap = plt.get_cmap("hsv")
    # Берем цвет, конвертируем в 0..255
    rgba = cmap(class_id / 28.0)  # 28 - примерно кол-во позвонков
    return tuple(int(x * 255) for x in rgba[:3])


def draw_polygons(image, label_path, class_names):
    """Рисует полигоны и подписи на изображении"""
    # Создаем копию для рисования
    annotated_img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_text = ImageDraw.Draw(annotated_img)

    w, h = image.size

    if not os.path.exists(label_path):
        return annotated_img, False

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return annotated_img, False

    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]

        # YOLO формат полигонов: id x1 y1 x2 y2 ...
        # Координаты нормализованы (0..1), нужно умножить на ширину/высоту
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * w
            y = coords[i + 1] * h
            points.append((x, y))

        if len(points) < 3:
            continue

        color = get_color(class_id)
        # Полупрозрачная заливка
        fill_color = color + (100,)  # 100 - альфа канал (прозрачность)
        outline_color = color + (255,)

        draw.polygon(points, fill=fill_color, outline=outline_color)

        # Рисуем текст (имя класса) в центре полигона
        # Находим среднюю точку
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        label_name = class_names.get(class_id, str(class_id))
        # Очищаем имя от "vertebrae " для краткости
        label_name = label_name.replace("vertebrae ", "").upper()

        draw_text.text(
            (center_x, center_y),
            label_name,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    # Склеиваем слои
    out = Image.alpha_composite(annotated_img, overlay)
    return out.convert("RGB"), True


# --- ИНТЕРФЕЙС STREAMLIT ---

st.title("🦴 Проверка качества разметки (YOLO Format)")

# 1. Загрузка классов
class_names = load_class_names()

# 2. Выбор выборки (Train/Val)
col_control1, col_control2 = st.columns([1, 3])
with col_control1:
    split = st.radio("Выберите папку:", ["train", "val"], horizontal=True)

img_dir = os.path.join(BASE_DIR, "images", split)
lbl_dir = os.path.join(BASE_DIR, "labels", split)

# Получаем список файлов
if os.path.exists(img_dir):
    all_images = sorted(glob(os.path.join(img_dir, "*.png")))
    # Оставляем только имена файлов для красоты списка
    img_names = [os.path.basename(p) for p in all_images]
else:
    st.error(f"Папка {img_dir} не найдена!")
    st.stop()

if not all_images:
    st.warning("В папке нет изображений.")
    st.stop()

# 3. Навигация
with col_control2:
    selected_file_name = st.selectbox("Выберите файл:", img_names)

# Индекс выбранного файла
current_idx = img_names.index(selected_file_name)
img_path = all_images[current_idx]
lbl_path = os.path.join(lbl_dir, selected_file_name.replace(".png", ".txt"))

# --- ОТРИСОВКА ---

col1, col2 = st.columns(2)

# Левая колонка: Оригинал
with col1:
    st.subheader("📸 Исходное изображение")
    try:
        image = Image.open(img_path).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"Файл: {os.path.basename(img_path)} | Размер: {image.size}")
    except Exception as e:
        st.error(f"Ошибка открытия: {e}")

# Правая колонка: С разметкой
with col2:
    st.subheader("🎯 С наложением масок")
    annotated_image, has_labels = draw_polygons(image, lbl_path, class_names)
    st.image(annotated_image, use_container_width=True)

    if has_labels:
        st.success("Разметка найдена и отрисована.")
    else:
        st.warning("⚠️ Файл разметки (.txt) пуст или не найден!")

# Доп. инфо: показать содержимое txt файла (если интересно)
with st.expander("Показать сырые данные разметки (YOLO txt)"):
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            st.text(f.read())
    else:
        st.write("Файл разметки отсутствует.")
