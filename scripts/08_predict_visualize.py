import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from ultralytics import YOLO

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

MODEL_PATH = Path("data/05_runs/spine_segmentation_v1/weights/best.pt")
DEFAULT_IMG_DIR = "data/04_normalized/test/images"

# ============================================================================
# ЛОГИКА ПРОСМОТРА
# ============================================================================

class DatasetViewer:
    def __init__(self, model, image_paths):
        self.model = model
        self.image_paths = sorted(image_paths)
        self.index = 0
        self.total = len(self.image_paths)

        # Создаем окно Matplotlib
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.2) # Оставляем место внизу для кнопок

        # Подключаем обработку клавиш (стрелки)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Создаем кнопки GUI
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next (->)')
        self.bprev = Button(axprev, 'Prev (<-)')
        
        self.bnext.on_clicked(self.next_image)
        self.bprev.on_clicked(self.prev_image)

        # Отображаем первое фото
        self.update_plot()
        plt.show()

    def on_key(self, event):
        """Обработка нажатий клавиатуры."""
        if event.key == 'right':
            self.next_image(None)
        elif event.key == 'left':
            self.prev_image(None)

    def next_image(self, event):
        self.index = (self.index + 1) % self.total
        self.update_plot()

    def prev_image(self, event):
        self.index = (self.index - 1) % self.total
        self.update_plot()

    def get_ground_truth(self, img_path, image):
        """Рисует зеленые полигоны из разметки."""
        # Ищем файл в папке labels (параллельно images)
        label_path = img_path.parents[1] / "labels" / f"{img_path.stem}.txt"
        
        gt_img = image.copy()
        if len(gt_img.shape) == 2:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
        else:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        if not label_path.exists():
            return None # Нет разметки

        h, w = gt_img.shape[:2]
        with open(label_path, 'r') as f:
            lines = f.readlines()

        found = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            found = True
            
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            points = coords.astype(np.int32)
            
            # Рисуем полигон
            cv2.polylines(gt_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
        return gt_img if found else None

    def update_plot(self):
        """Основная функция отрисовки."""
        img_path = self.image_paths[self.index]
        
        # 1. Чтение
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"Error reading {img_path}")
            return

        # 2. Предикт
        # boxes=False убирает рамки
        # retina_masks=True делает маски красивыми
        # Убрал параметр alpha, так как он вызвал ошибку
        results = self.model(original_img, retina_masks=True, verbose=False)
        pred_plot = results[0].plot(boxes=False, conf=True)
        pred_rgb = cv2.cvtColor(pred_plot, cv2.COLOR_BGR2RGB)

        # 3. Ground Truth
        gt_rgb = self.get_ground_truth(img_path, original_img)

        # 4. Подготовка оригинала для вывода
        if len(original_img.shape) == 2:
            orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # 5. Отрисовка в Matplotlib
        
        # Очищаем старые картинки
        for ax in self.axes:
            ax.clear()
            ax.axis('off')

        # Заголовок окна
        self.fig.canvas.manager.set_window_title(f"Image {self.index + 1}/{self.total}: {img_path.name}")
        self.fig.suptitle(f"File: {img_path.name} [{self.index + 1}/{self.total}]", fontsize=14)

        # Рисуем Оригинал
        self.axes[0].imshow(orig_rgb)
        self.axes[0].set_title("Оригинал")

        # Рисуем Разметку (если есть) или пустоту
        if gt_rgb is not None:
            self.axes[1].imshow(gt_rgb)
            self.axes[1].set_title("Разметка (Manual GT)")
        else:
            self.axes[1].text(0.5, 0.5, "Нет файла разметки", 
                              ha='center', va='center', fontsize=12)
            self.axes[1].set_title("Разметка отсутствует")

        # Рисуем Предикт
        self.axes[2].imshow(pred_rgb)
        self.axes[2].set_title("Результат Модели")

        # Обновляем холст
        self.fig.canvas.draw_idle()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not MODEL_PATH.exists():
        print(f"ОШИБКА: Модель не найдена по пути: {MODEL_PATH}")
        return

    # 1. Выбор папки
    root = tk.Tk()
    root.withdraw()
    print("Выберите ПАПКУ с изображениями...")
    
    selected_dir = filedialog.askdirectory(
        initialdir=DEFAULT_IMG_DIR if Path(DEFAULT_IMG_DIR).exists() else ".",
        title="Выберите папку с изображениями (images)"
    )

    if not selected_dir:
        print("Папка не выбрана.")
        return

    # 2. Поиск изображений
    img_dir = Path(selected_dir)
    extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.png"}
    images = []
    for ext in extensions:
        images.extend(list(img_dir.glob(ext)))
    
    if not images:
        print(f"В папке {selected_dir} изображений не найдено.")
        return

    print(f"Найдено {len(images)} изображений. Загрузка модели...")
    
    # 3. Загрузка модели
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Ошибка модели: {e}")
        return

    # 4. Запуск просмотрщика
    print("Запуск интерфейса...")
    viewer = DatasetViewer(model, images)

if __name__ == "__main__":
    main()