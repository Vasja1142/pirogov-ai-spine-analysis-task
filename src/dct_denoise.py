import streamlit as st
import numpy as np
import cv2

st.set_page_config(layout="wide", page_title="PRO: PRE-CLAHE FILLING")

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---


def to_uint8_display(img_data):
    if img_data is None:
        return None
    if img_data.dtype == np.uint8:
        return img_data
    return np.clip(img_data, 0, 255).astype(np.uint8)


def safe_load_opencv(file_bytes):
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if len(raw.shape) == 3:
        if raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if norm.dtype != np.uint8:
        norm = norm.astype(np.uint8)
    return norm


def apply_multipass_clahe(img, clip_limit, grid_size, passes=1):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if passes <= 1:
        return clahe.apply(img)
    h, w = img.shape
    accumulator = np.zeros((h, w), dtype=np.float32)
    for i in range(passes):
        dy = int((grid_size * i) / passes)
        dx = int((grid_size * i) / passes)
        padded = cv2.copyMakeBorder(img, dy, 0, dx, 0, cv2.BORDER_REFLECT)
        res = clahe.apply(padded)
        accumulator += res[dy : dy + h, dx : dx + w].astype(np.float32)
    return np.clip(accumulator / passes, 0, 255).astype(np.uint8)


def apply_progressive_hole_filling(img, max_k=5):
    """
    Прогрессивное заполнение (3, 4, 5, 6...).
    Применяется ДО CLAHE, чтобы восстановить структуру.
    """
    result = img.copy()

    # Идем с шагом 1 (3, 4, 5...)
    for k in range(3, max_k + 1):
        # 1. Создаем ядра
        ring_kernel = np.ones((k, k), np.uint8)
        inner_pad = 1
        ring_kernel[inner_pad:-inner_pad, inner_pad:-inner_pad] = 0

        inner_kernel = np.zeros((k, k), np.uint8)
        inner_kernel[inner_pad:-inner_pad, inner_pad:-inner_pad] = 1

        # 2. Вычисляем экстремумы
        min_ring = cv2.erode(result, ring_kernel)
        max_ring = cv2.dilate(result, ring_kernel)
        max_inner = cv2.dilate(result, inner_kernel)
        min_inner = cv2.erode(result, inner_kernel)

        # 3. Логика
        # Яма: Вся внутренность ниже минимума забора
        is_pit = max_inner < min_ring
        result[is_pit] = min_ring[is_pit]

        # Холм: Вся внутренность выше максимума забора
        is_hill = min_inner > max_ring
        result[is_hill] = max_ring[is_hill]

    return result


# --- ЯДРО ОБРАБОТКИ ---


def process_pipeline(
    img_input_uint8,
    # 1. PROGRESSIVE FILLING (Moved to Start)
    use_fill,
    fill_max_k,
    # 2. CLAHE
    use_clahe,
    clahe_clip,
    clahe_grid,
    clahe_passes,
    # 3. Bilateral
    use_bilat,
    bilat_d,
    bilat_sc,
    bilat_ss,
    # 4. NLM
    use_nlm,
    nlm_h,
    nlm_t,
    nlm_s,
    # 5. SMART USM
    use_usm,
    usm_amt,
    usm_sigma,
    usm_smart_blur,
    # 6. SMART LAPLACIAN
    use_lap,
    lap_str,
    lap_blur,
    # 7. SSAA
    use_ssaa,
    ssaa_factor,
    ssaa_smooth,
):
    results = {}
    results["original"] = img_input_uint8.copy()
    current_uint8 = img_input_uint8

    # --- ЭТАП 1: PROGRESSIVE FILLING (Лечим структуру) ---
    if use_fill:
        current_uint8 = apply_progressive_hole_filling(current_uint8, max_k=fill_max_k)
    results["after_fill"] = current_uint8.copy()

    # --- ЭТАП 2: CLAHE (Усиливаем контраст) ---
    if use_clahe:
        current_uint8 = apply_multipass_clahe(
            current_uint8, clahe_clip, clahe_grid, clahe_passes
        )
    results["after_clahe"] = current_uint8.copy()

    # --> Float32
    current_float = current_uint8.astype(np.float32)

    # --- ЭТАП 3: Bilateral ---
    if use_bilat and bilat_d != 0:
        current_float = cv2.bilateralFilter(
            current_float, d=bilat_d, sigmaColor=bilat_sc, sigmaSpace=bilat_ss
        )
    results["after_bilateral"] = current_float.copy()

    # --- ЭТАП 4: NLM ---
    if use_nlm and nlm_h > 0:
        temp_uint8 = np.clip(current_float, 0, 255).astype(np.uint8)
        t = int(nlm_t) | 1
        s = int(nlm_s) | 1
        denoised = cv2.fastNlMeansDenoising(temp_uint8, None, nlm_h, t, s)
        current_float = denoised.astype(np.float32)
    results["after_nlm"] = current_float.copy()

    # --- ЭТАП 5: SMART USM ---
    if use_usm and usm_amt > 0:
        if usm_smart_blur > 0:
            mask_src = cv2.GaussianBlur(current_float, (0, 0), sigmaX=usm_smart_blur)
        else:
            mask_src = current_float
        usm_blur = cv2.GaussianBlur(mask_src, (0, 0), sigmaX=usm_sigma)
        edge_mask = cv2.addWeighted(mask_src, 1.0, usm_blur, -1.0, 0)
        current_float = cv2.addWeighted(current_float, 1.0, edge_mask, usm_amt, 0)
    results["after_usm"] = current_float.copy()

    # --- ЭТАП 6: SMART LAPLACIAN ---
    if use_lap and lap_str > 0:
        if lap_blur > 0:
            detector = cv2.GaussianBlur(current_float, (0, 0), sigmaX=lap_blur)
        else:
            detector = current_float
        lap = cv2.Laplacian(detector, cv2.CV_32F, ksize=1)
        current_float = current_float - (lap * lap_str)

    results["before_ssaa"] = current_float.copy()

    # --- ЭТАП 7: SSAA ---
    if use_ssaa:
        h, w = current_float.shape[:2]
        up_h, up_w = int(h * ssaa_factor), int(w * ssaa_factor)
        upscaled = cv2.resize(
            current_float, (up_w, up_h), interpolation=cv2.INTER_CUBIC
        )
        if ssaa_smooth > 0:
            upscaled = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=ssaa_smooth)
        current_float = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_AREA)

    results["final"] = current_float
    return results


# --- ИНТЕРФЕЙС ---

st.title("💎 PRO: PRE-CLAHE RESTORATION")
st.markdown(
    "**Prog. Filler** -> **CLAHE** -> **Bilateral** -> **NLM** -> **Sharpen** -> **SSAA**"
)

with st.sidebar:
    uploaded_file = st.file_uploader("Файл", type=["jpg", "png", "tif", "bmp"])
    st.markdown("---")

    # 1. PROGRESSIVE FILLER (ПЕРВЫМ!)
    with st.expander("1. Progressive Filler (Восстановление)", expanded=True):
        st.info("Применяется к сырому изображению. Чистит структуру до контраста.")
        use_fill = st.checkbox("Вкл Filler", value=True)
        p_fill_max = st.slider(
            "Макс размер окна",
            3,
            10,
            5,
            help="3=чистит точки, 4=чистит 2x2, 5=чистит 3x3...",
        )

    # 2. CLAHE
    with st.expander("2. Multipass CLAHE", expanded=True):
        use_clahe = st.checkbox("Вкл CLAHE", value=True)
        col1, col2 = st.columns(2)
        with col1:
            p_clahe_clip = st.slider("Clip Limit", 0.5, 10.0, 2.0, step=0.1)
        with col2:
            p_clahe_grid = st.select_slider("Grid", [4, 8, 12, 16, 24, 32], value=8)
        p_clahe_passes = st.slider("Passes", 1, 16, 8)

    # 3. BILATERAL
    with st.expander("3. Bilateral (Float)", expanded=True):
        use_bilat = st.checkbox("Вкл Bilateral", value=True)
        p_bilat_d = st.slider("Diameter", 0, 25, 5)
        p_bilat_sc = st.slider("Sigma Color", 0, 200, 75)
        p_bilat_ss = st.slider("Sigma Space", 0.1, 10.0, 5.0)

    # 4. NLM
    with st.expander("4. NLM (Cleaner)", expanded=True):
        use_nlm = st.checkbox("Вкл NLM", value=True)
        p_nlm_h = st.slider("Strength (h)", 1, 50, 10)
        p_nlm_t = st.slider("Template", 3, 21, 7, step=2)
        p_nlm_s = st.slider("Search", 3, 45, 21, step=2)

    # 5. SHARPNESS
    with st.expander("5. Резкость", expanded=True):
        use_usm = st.checkbox("Вкл USM", value=True)
        p_usm_amt = st.slider("USM Amount", 0.0, 10.0, 1.5, step=0.1)
        p_usm_sig = st.slider("USM Radius", 0.1, 10.0, 2.0, step=0.1)
        p_usm_smart = st.slider("USM Smart Blur", 0.0, 5.0, 1.0)
        st.markdown("---")
        use_lap = st.checkbox("Вкл Laplacian", value=True)
        p_lap_str = st.slider("Lap Strength", 0.0, 10.0, 0.5, step=0.05)
        p_lap_blur = st.slider("Lap Smart Blur", 0.0, 5.0, 1.0)

    # 6. SSAA
    with st.expander("6. SSAA (Антиалиасинг)", expanded=True):
        use_ssaa = st.checkbox("Вкл SSAA", value=True)
        p_ssaa_factor = st.select_slider(
            "Upscale Factor", options=[1.5, 2.0, 3.0, 4.0], value=2.0
        )
        p_ssaa_smooth = st.slider("High-Res Smooth", 0.0, 5.0, 1.5, step=0.1)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    base_img = safe_load_opencv(file_bytes)

    if base_img is not None:
        data = process_pipeline(
            base_img,
            use_fill,
            p_fill_max,  # Filler First
            use_clahe,
            p_clahe_clip,
            p_clahe_grid,
            p_clahe_passes,
            use_bilat,
            p_bilat_d,
            p_bilat_sc,
            p_bilat_ss,
            use_nlm,
            p_nlm_h,
            p_nlm_t,
            p_nlm_s,
            use_usm,
            p_usm_amt,
            p_usm_sig,
            p_usm_smart,
            use_lap,
            p_lap_str,
            p_lap_blur,
            use_ssaa,
            p_ssaa_factor,
            p_ssaa_smooth,
        )

        st.subheader("1. Чистка сырых данных")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                to_uint8_display(data["original"]),
                caption="Оригинал (Сырой)",
                use_container_width=True,
            )
        with c2:
            st.image(
                to_uint8_display(data["after_fill"]),
                caption=f"После Filler (До контраста)",
                use_container_width=True,
            )

        st.subheader("2. Результат обработки")
        c3, c4 = st.columns(2)
        with c3:
            st.image(
                to_uint8_display(data["after_clahe"]),
                caption="После CLAHE",
                use_container_width=True,
            )
        with c4:
            st.image(
                to_uint8_display(data["final"]),
                caption="ФИНАЛ",
                use_container_width=True,
            )
