import streamlit as st
import numpy as np
import cv2

st.set_page_config(layout="wide", page_title="PRO SHARPENER")

# --- ФУНКЦИИ (Тот самый рабочий вариант) ---


def safe_load_opencv(file_bytes):
    """
    Функция из версии, которая у тебя заработала.
    """
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    # Цвет -> ЧБ
    if len(raw.shape) == 3:
        if raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw

    # Нормализация (лечит черноту)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Гарантируем uint8
    if norm.dtype != np.uint8:
        norm = norm.astype(np.uint8)

    return norm


def apply_pipeline(img, nlm_h, nlm_t, nlm_s, usm_amt, usm_rad, lap_str):
    """
    Весь конвейер в одной функции с защитой от ошибок.
    """
    # 1. NLM (Очистка) - работает с uint8
    if nlm_h > 0:
        # Принудительно нечетные окна (требование OpenCV)
        t = int(nlm_t) | 1
        s = int(nlm_s) | 1
        denoised = cv2.fastNlMeansDenoising(img, None, nlm_h, t, s)
    else:
        denoised = img

    # Дальше работаем во float64 для точности
    work_img = denoised.astype(np.float64)

    # 2. USM (Резкость 1-го порядка) - Контраст
    if usm_amt > 0:
        blur = cv2.GaussianBlur(work_img, (0, 0), sigmaX=usm_rad)
        # Формула: Оригинал + (Оригинал - Размытие) * Сила
        work_img = cv2.addWeighted(work_img, 1.0 + usm_amt, blur, -usm_amt, 0)

    # 3. Laplacian (Резкость 2-го порядка) - Детали
    if lap_str > 0:
        # Вычисляем вторую производную
        lap = cv2.Laplacian(work_img, cv2.CV_64F, ksize=1)
        # Вычитаем её (так работает повышение резкости через Лапласиан)
        work_img = work_img - (lap * lap_str)

    # 4. Финал: Обрезаем всё, что вылезло за 0..255
    final = np.clip(work_img, 0, 255).astype(np.uint8)

    return denoised, final


# --- ИНТЕРФЕЙС ---

st.title("🔥 BARE METAL + LAPLACIAN")
st.markdown("Рабочее ядро + Усилитель деталей.")

with st.sidebar:
    uploaded_file = st.file_uploader("Файл", type=["jpg", "png", "tif", "bmp"])

    st.header("1. Очистка (NLM)")
    p_h = st.slider("Сила мыла", 0, 50, 10)
    p_t = st.slider("Патч", 3, 31, 7, step=2)
    p_s = st.slider("Поиск", 11, 45, 21, step=2)

    st.divider()

    st.header("2. Резкость (Контуры)")
    st.info("Производная 1-го порядка (USM)")
    u_amt = st.slider("Сила (Amount)", 0.0, 10.0, 1.5, 0.1)
    u_rad = st.slider("Толщина (Radius)", 0.5, 20.0, 2.0, 0.5)

    st.divider()

    st.header("3. Детали (Хруст)")
    st.info("Производная 2-го порядка (Laplacian). Осторожно!")
    l_str = st.slider(
        "Микро-резкость", 0.0, 10.0, 0.0, 0.05, help="Добавляет 'звон' и мелкие детали."
    )

if uploaded_file:
    # 1. Загрузка
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    base_img = safe_load_opencv(file_bytes)

    if base_img is None:
        st.error("Файл не читается.")
    else:
        # 2. Обработка с отловом ошибок
        try:
            # Оптимизация превью
            h, w = base_img.shape
            if h > 2500:
                scale = 2500 / h
                proc_in = cv2.resize(base_img, None, fx=scale, fy=scale)
            else:
                proc_in = base_img

            # ЗАПУСК КОНВЕЙЕРА
            img_denoised, img_final = apply_pipeline(
                proc_in, p_h, p_t, p_s, u_amt, u_rad, l_str
            )

            # 3. Вывод
            st.write("### Результат")

            # Зум
            col1, col2 = st.columns(2)

            crop = 250
            h_f, w_f = img_final.shape
            cy, cx = h_f // 2, w_f // 2
            y1, y2 = max(0, cy - crop), min(h_f, cy + crop)
            x1, x2 = max(0, cx - crop), min(w_f, cx + crop)

            col1.image(
                img_denoised[y1:y2, x1:x2],
                caption="Только очистка (NLM)",
                use_container_width=True,
            )
            col2.image(
                img_final[y1:y2, x1:x2],
                caption="Финал (USM + Лаплас)",
                use_container_width=True,
            )

            # Полная картинка
            st.image(img_final, caption="Полное изображение", use_container_width=True)

            # Скачивание
            res_bytes = cv2.imencode(".png", img_final)[1].tobytes()
            st.download_button(
                "Скачать PNG", res_bytes, "ultra_sharp.png", "image/png", type="primary"
            )

        except Exception as e:
            st.error(f"Ошибка алгоритма: {e}")
            st.image(base_img, caption="Показываю оригинал, так как обработка упала.")
