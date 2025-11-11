"""
GUI-приложение для анализа и визуализации эффектов нормализации и аугментации.

Возможности:
- Интерактивная настройка параметров нормализации (CLAHE, фильтры, гамма-коррекция)
- Визуализация различных аугментаций (наклон, эластичные деформации, шум Перлина)
- Отображение результата LoG (Laplacian of Gaussian) для анализа границ
- Режим реального времени с обновлением при изменении параметров
"""

import math
import random
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image, ImageTk
from skimage.restoration import denoise_tv_chambolle


# ============================================================================
# ОСНОВНОЙ КЛАСС ПРИЛОЖЕНИЯ
# ============================================================================

class VisualAnalyzer(tk.Tk):
    """Главное окно приложения для визуального анализа."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Анализатор нормализаций и аугментаций")
        self.geometry("1600x900")

        # Переменные состояния
        self.original_img = None
        self.processed_img = None
        self.img_path = None
        self.is_log_slider = False

        # Создание интерфейса
        self._setup_ui()
        
        # Автоматическая загрузка изображения по умолчанию
        default_image_path = (
            "/home/vasja1142/Рабочий стол/pirogov-ai-spine-analysis-task/"
            "data/02_processed/Vertebrae Detection.v2i.yolov12/test/images/"
            "Vertebrae Detection.v2i.yolov12_N2-Lt-T-and-Rt-L-AIS-M-17-yrs_"
            "png.rf.56fa904adb4b00a3ebeddf6c47a6fd3a.jpg"
        )
        self.load_image(default_image_path)

    def _setup_ui(self):
        """Создает структуру пользовательского интерфейса."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Панель управления (слева)
        control_panel = ttk.Frame(main_frame, width=450)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        # Панель изображений (справа)
        image_panel = ttk.Frame(main_frame)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Кнопка загрузки изображения
        load_button = ttk.Button(
            control_panel,
            text="Загрузить изображение",
            command=self.load_image
        )
        load_button.pack(pady=10, fill=tk.X, padx=5)

        # Вкладки для различных настроек
        self.notebook = ttk.Notebook(control_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5)

        self.norm_tab = ttk.Frame(self.notebook)
        self.aug_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.norm_tab, text="Нормализация")
        self.notebook.add(self.aug_tab, text="Аугментация")

        self._create_norm_widgets()
        self._create_aug_widgets()
        self._create_log_widgets(control_panel)

        # Холсты для отображения изображений
        self.canvas_orig = self._create_image_canvas(image_panel, "Оригинал")
        self.canvas_proc = self._create_image_canvas(image_panel, "Обработанное")
        self.canvas_diff = self._create_image_canvas(image_panel, "LoG (Края)")

    def _create_image_canvas(self, parent, title):
        """Создает холст для отображения изображения."""
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas = tk.Canvas(frame, bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    # ========================================================================
    # СОЗДАНИЕ ВИДЖЕТОВ НОРМАЛИЗАЦИИ
    # ========================================================================

    def _create_norm_widgets(self):
        """Создает виджеты для настройки параметров нормализации."""
        
        # CLAHE
        clahe_frame = ttk.LabelFrame(self.norm_tab, text="CLAHE")
        clahe_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.clahe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            clahe_frame,
            text="Применить CLAHE",
            variable=self.clahe_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.clahe_clip = self._create_slider(
            clahe_frame, "Clip Limit", 1, 40, 2
        )
        self.clahe_grid = self._create_slider(
            clahe_frame, "Grid Size", 1, 256, 8
        )

        # Фильтрация
        blur_frame = ttk.LabelFrame(self.norm_tab, text="Фильтрация")
        blur_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.median_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            blur_frame,
            text="Применить Median Blur",
            variable=self.median_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.median_k = self._create_slider(
            blur_frame, "Median Blur Kernel", 1, 21, 3, is_odd=True
        )
        
        self.bilateral_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            blur_frame,
            text="Применить Bilateral Filter",
            variable=self.bilateral_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.bilateral_d = self._create_slider(
            blur_frame, "Bilateral Diameter", 1, 50, 9
        )
        self.bilateral_sc = self._create_slider(
            blur_frame, "Bilateral Sigma Color", 1, 1500, 75
        )
        self.bilateral_ss = self._create_slider(
            blur_frame, "Bilateral Sigma Space", 0.1, 5, 75
        )

        # Другие методы
        other_frame = ttk.LabelFrame(self.norm_tab, text="Другие методы")
        other_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.gamma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            other_frame,
            text="Применить Gamma Correction",
            variable=self.gamma_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.gamma = self._create_slider(
            other_frame, "Gamma", 0.1, 3.0, 1.0, resolution=0.1
        )
        
        self.ad_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            other_frame,
            text="Применить Anisotropic Diffusion",
            variable=self.ad_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.zscore_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            other_frame,
            text="Применить Z-Score Normalize",
            variable=self.zscore_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)

    # ========================================================================
    # СОЗДАНИЕ ВИДЖЕТОВ АУГМЕНТАЦИИ
    # ========================================================================

    def _create_aug_widgets(self):
        """Создает виджеты для настройки параметров аугментации."""
        
        # Наклон (Shear)
        shear_frame = ttk.LabelFrame(self.aug_tab, text="Наклон (Shear)")
        shear_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.shear_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            shear_frame,
            text="Применить Наклон",
            variable=self.shear_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.shear = self._create_slider(
            shear_frame, "Shear Angle", -45, 45, 0
        )

        # Эластичные деформации
        elastic_frame = ttk.LabelFrame(
            self.aug_tab,
            text="Эластичные деформации"
        )
        elastic_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.elastic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            elastic_frame,
            text="Применить Эластичные деформации",
            variable=self.elastic_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.elastic_alpha = self._create_slider(
            elastic_frame, "Alpha (Интенсивность)", 0, 1500, 120
        )
        self.elastic_sigma = self._create_slider(
            elastic_frame, "Sigma (Масштаб)", 1, 5000, 20
        )
        self.elastic_affine = self._create_slider(
            elastic_frame, "Affine Component", 0, 200, 50
        )

        # Шум Перлина
        perlin_frame = ttk.LabelFrame(
            self.aug_tab,
            text="Шум Перлина (множитель)"
        )
        perlin_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.perlin_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            perlin_frame,
            text="Применить Шум Перлина",
            variable=self.perlin_var,
            command=self.apply_transforms
        ).pack(anchor=tk.W)
        
        self.perlin_scale = self._create_slider(
            perlin_frame, "Масштаб шума", 1, 512, 8
        )
        self.perlin_intensity = self._create_slider(
            perlin_frame, "Интенсивность", 0.0, 1.0, 0.5, resolution=0.05
        )

    # ========================================================================
    # СОЗДАНИЕ ВИДЖЕТОВ LoG
    # ========================================================================

    def _create_log_widgets(self, parent):
        """Создает панель с настройками для изображения краев (LoG)."""
        log_frame = ttk.LabelFrame(
            parent,
            text="Настройки отображения краев (LoG)"
        )
        log_frame.pack(pady=10, padx=5, fill=tk.X)

        self.log_blur_k = self._create_slider(
            log_frame, "Размер ядра размытия", 1, 15, 3,
            is_odd=True, is_log=True
        )
        self.log_clahe_clip = self._create_slider(
            log_frame, "CLAHE Clip Limit", 1.0, 40.0, 2.0,
            resolution=0.1, is_log=True
        )
        self.log_clahe_grid = self._create_slider(
            log_frame, "CLAHE Grid Size", 1, 64, 8, is_log=True
        )

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ СОЗДАНИЯ СЛАЙДЕРОВ
    # ========================================================================

    def _create_slider(
        self,
        parent,
        text,
        from_,
        to,
        default,
        resolution=1,
        is_odd=False,
        is_log=False
    ):
        """Создает слайдер с меткой и отображением значения."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        label = ttk.Label(frame, text=text, width=22)
        label.pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        
        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda s, v=var, o=is_odd, log=is_log:
                self._on_slider_change(s, v, o, log)
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        value_label = ttk.Label(frame, text=f"{default:.2f}", width=6)
        value_label.pack(side=tk.RIGHT)
        
        var.trace_add(
            "write",
            lambda *args, v=var, l=value_label:
                l.config(text=f"{v.get():.2f}")
        )
        
        return var

    def _on_slider_change(self, value_str, var, is_odd, is_log):
        """Обрабатывает изменение значения слайдера."""
        value = float(value_str)
        
        if is_odd:
            value = int(round(value))
            if value % 2 == 0:
                value += 1
            var.set(value)
        
        # Обновляем только то, что нужно
        if is_log:
            self.calculate_and_display_diff()
        else:
            self.apply_transforms()

    # ========================================================================
    # МЕТОДЫ ЗАГРУЗКИ И ОТОБРАЖЕНИЯ ИЗОБРАЖЕНИЙ
    # ========================================================================

    def load_image(self, default_path=None):
        """Загружает изображение для анализа."""
        if default_path:
            path = default_path
        else:
            path = filedialog.askopenfilename(
                initialdir="data/02_processed",
                title="Выберите изображение",
                filetypes=(
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                )
            )
        
        if not path:
            return
        
        self.img_path = path
        self.original_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.display_image(self.original_img, self.canvas_orig)
        self.apply_transforms()

    def display_image(self, img_cv, canvas):
        """Отображает изображение на холсте."""
        if img_cv is None:
            return
        
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            canvas_w, canvas_h = 500, 800
        
        h, w = img_cv.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_img = cv2.resize(
            img_cv,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        img_pil = Image.fromarray(resized_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        canvas.delete("all")
        canvas.create_image(
            canvas_w / 2,
            canvas_h / 2,
            anchor=tk.CENTER,
            image=img_tk
        )
        canvas.image = img_tk

    # ========================================================================
    # ПРИМЕНЕНИЕ ТРАНСФОРМАЦИЙ
    # ========================================================================

    def apply_transforms(self):
        """Применяет выбранные трансформации к изображению."""
        if self.original_img is None:
            return

        img = self.original_img.copy()
        active_tab = self.notebook.tab(self.notebook.select(), "text")

        if active_tab == "Нормализация":
            img = self._apply_normalization(img)
        elif active_tab == "Аугментация":
            img = self._apply_augmentation(img)

        self.processed_img = img
        self.display_image(self.processed_img, self.canvas_proc)
        self.calculate_and_display_diff()

    def _apply_normalization(self, img):
        """Применяет нормализации к изображению."""
        if self.clahe_var.get():
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip.get(),
                tileGridSize=(
                    int(self.clahe_grid.get()),
                    int(self.clahe_grid.get())
                )
            )
            img = clahe.apply(img)

        if self.bilateral_var.get():
            d = int(self.bilateral_d.get())
            sc = self.bilateral_sc.get()
            ss = self.bilateral_ss.get()
            img = cv2.bilateralFilter(img, d, sc, ss)

        if self.median_var.get():
            ksize = int(self.median_k.get())
            img = cv2.medianBlur(img, ksize)

        if self.gamma_var.get():
            gamma = self.gamma.get()
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            img = cv2.LUT(img, table)

        if self.ad_var.get():
            img_float = img.astype(np.float32) / 255.0
            img = denoise_tv_chambolle(
                img_float,
                weight=0.1,
                multichannel=False
            )
            img = (img * 255).astype(np.uint8)

        if self.zscore_var.get():
            mean, std = cv2.meanStdDev(img)
            if std[0] > 1e-6:
                img = (img - mean[0]) / std[0]
            img = cv2.normalize(
                img, None, 0, 255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
        
        return img

    def _apply_augmentation(self, img):
        """Применяет аугментации к изображению."""
        rows, cols = img.shape

        if self.shear_var.get():
            shear_angle = self.shear.get()
            if shear_angle != 0:
                M_shear = np.float32([
                    [1, math.tan(math.radians(shear_angle)), 0],
                    [0, 1, 0]
                ])
                img = cv2.warpAffine(img, M_shear, (cols, rows))

        if self.elastic_var.get():
            alpha = self.elastic_alpha.get()
            sigma = self.elastic_sigma.get()
            
            if alpha > 0 and sigma > 0:
                random_state = np.random.RandomState(None)
                
                dx = cv2.GaussianBlur(
                    (random_state.rand(rows, cols) * 2 - 1),
                    (int(sigma)|1, int(sigma)|1),
                    0
                ) * alpha
                
                dy = cv2.GaussianBlur(
                    (random_state.rand(rows, cols) * 2 - 1),
                    (int(sigma)|1, int(sigma)|1),
                    0
                ) * alpha
                
                x, y = np.meshgrid(np.arange(cols), np.arange(rows))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)
                
                img = cv2.remap(
                    img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR
                )

        if self.perlin_var.get():
            intensity = self.perlin_intensity.get()
            
            if intensity > 0:
                scale = self.perlin_scale.get()
                noise = PerlinNoise(octaves=scale)
                
                pic = np.array([
                    [noise([i/cols, j/rows]) for j in range(cols)]
                    for i in range(rows)
                ])
                
                multiplier = 1 + (pic * intensity)
                img = np.clip(img * multiplier, 0, 255).astype(np.uint8)

        return img

    # ========================================================================
    # РАСЧЕТ И ОТОБРАЖЕНИЕ LoG
    # ========================================================================

    def calculate_and_display_diff(self):
        """Вычисляет и отображает изображение LoG (границы)."""
        if self.processed_img is None:
            self.canvas_diff.delete("all")
            return

        # Получение динамических параметров
        blur_ksize = int(self.log_blur_k.get())
        clahe_clip = self.log_clahe_clip.get()
        clahe_grid = int(self.log_clahe_grid.get())

        # Применение LoG
        blurred = cv2.GaussianBlur(
            self.processed_img,
            (blur_ksize, blur_ksize),
            0
        )
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # Преобразование в 8-битное изображение
        log_image = np.uint8(np.absolute(laplacian))

        # Нормализация
        if log_image.max() > 0:
            log_image = cv2.normalize(
                log_image, None, 0, 255,
                cv2.NORM_MINMAX
            )
        
        # Применение CLAHE для усиления деталей
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid)
        )
        bright_log = clahe.apply(log_image)
        
        self.display_image(bright_log, self.canvas_diff)


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    app = VisualAnalyzer()
    app.mainloop()