import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2

# --- НАСТРОЙКИ ---
MODEL_PATH = "data/05_run/run_data_enhanced_only_clahe/weights/best.pt"
TEST_IMAGES_DIR = "test_images"

st.set_page_config(layout="wide", page_title="YOLOv11 Spine Inference (CLAHE)")


# --- ФУНКЦИЯ CLAHE (ТОЧНО КАК ПРИ ОБУЧЕНИИ) ---
def apply_multipass_clahe(img, clip_limit=4.50, grid_size=24, passes=8):
    """
    Применяет многопроходный CLAHE к изображению.
    Вход: Grayscale изображение (numpy array)
    """
    # Защита: если пришло цветное, делаем ЧБ
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    if passes <= 1:
        return clahe.apply(img)

    h, w = img.shape
    accumulator = np.zeros((h, w), dtype=np.float32)

    for i in range(passes):
        # Сдвиг сетки для устранения артефактов
        dy = int((grid_size * i) / passes)
        dx = int((grid_size * i) / passes)
        padded = cv2.copyMakeBorder(img, dy, 0, dx, 0, cv2.BORDER_REFLECT)
        res = clahe.apply(padded)
        accumulator += res[dy : dy + h, dx : dx + w].astype(np.float32)

    return np.clip(accumulator / passes, 0, 255).astype(np.uint8)


# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model(path):
    return YOLO(path)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()


# --- ОТРИСОВКА ---
def draw_predictions(image, results, conf_threshold, show_junk):
    out_img = image.convert("RGBA")
    overlay = Image.new("RGBA", out_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_text = ImageDraw.Draw(out_img)
    names = results.names

    if results.masks is not None:
        for i, mask in enumerate(results.masks.xy):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            class_name = names[cls_id]

            if conf < conf_threshold:
                continue

            if not show_junk:
                if "vertebrae" not in class_name.lower():
                    continue

            cmap = plt.get_cmap("hsv")
            color_rgb = cmap(cls_id / len(names))[:3]
            color_int = tuple(int(c * 255) for c in color_rgb)

            if len(mask) > 0:
                polygon = [tuple(point) for point in mask]
                fill_color = color_int + (90,)
                outline_color = color_int + (255,)

                draw.polygon(polygon, fill=fill_color, outline=outline_color)

                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                short_name = class_name.replace("vertebrae ", "").upper()

                # Рисуем текст с обводкой
                draw_text.text(
                    (center_x, center_y),
                    short_name,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )

    return Image.alpha_composite(out_img, overlay)


# --- ИНТЕРФЕЙС ---
st.title("🧠 Тестирование (Pre-processing: CLAHE)")

with st.sidebar:
    st.header("Настройки")
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(TEST_IMAGES_DIR, ext)))
    image_files = sorted(image_files)

    if not image_files:
        st.error(f"Нет изображений в папке {TEST_IMAGES_DIR}!")
        st.stop()

    img_names = [os.path.basename(p) for p in image_files]
    selected_name = st.selectbox("Выберите снимок:", img_names)

    conf_thresh = st.slider("Порог уверенности", 0.1, 1.0, 0.4, 0.05)
    show_junk = st.checkbox("Показывать все классы", value=False)

    st.info(
        "ℹ️ Перед подачей в нейросеть применяется Multipass CLAHE (как при обучении)."
    )

# --- ЛОГИКА ОБРАБОТКИ ---
img_path = os.path.join(TEST_IMAGES_DIR, selected_name)
original_pil = Image.open(img_path).convert("RGB")

# 1. Конвертация в OpenCV (numpy)
img_cv = np.array(original_pil)

# 2. ПРИМЕНЕНИЕ CLAHE
# Превращаем в Grayscale для обработки
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
# Применяем фильтр с вашими параметрами
processed_cv = apply_multipass_clahe(img_gray, clip_limit=4.50, grid_size=24, passes=8)

# 3. Возвращаем в RGB (для PIL и модели)
# YOLO ожидает 3 канала, поэтому дублируем чб канал 3 раза
processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_GRAY2RGB)
processed_pil = Image.fromarray(processed_rgb)

# 4. ИНФЕРЕНС (на обработанном изображении)
results = model.predict(processed_pil, conf=conf_thresh, imgsz=640)[0]

# 5. ОТРИСОВКА (поверх обработанного изображения)
result_image = draw_predictions(processed_pil, results, conf_thresh, show_junk)

# --- ВЫВОД НА ЭКРАН ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Вход в нейросеть (CLAHE)")
    st.image(
        processed_pil,
        caption="Clip: 4.5, Grid: 24, Passes: 8",
        use_container_width=True,
    )

with col2:
    st.subheader("Результат")
    st.image(result_image, caption="Предикт YOLOv11", use_container_width=True)

with st.expander("📊 Детальная статистика"):
    data = []
    if results.boxes:
        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            name = results.names[cls_id]
            if not show_junk and "vertebrae" not in name.lower():
                continue
            if conf < conf_thresh:
                continue
            data.append({"Class": name, "Confidence": f"{conf:.2f}"})

    if data:
        st.table(data)
    else:
        st.write("Ничего не найдено.")
