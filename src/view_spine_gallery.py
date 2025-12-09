import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from glob import glob
from PIL import Image

# --- НАСТРОЙКИ ---
output_dir = "output_res_2"
input_image_dir = "images"

# 1. Собираем список всех пациентов
subfolders = sorted([f.path for f in os.scandir(output_dir) if f.is_dir()])

if not subfolders:
    print("❌ Нет результатов.")
    exit()

print(f"📂 Найдено пациентов: {len(subfolders)}")

# Глобальный индекс текущего пациента
current_idx = 0

# Подготовка фигуры
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(bottom=0.2)  # Оставляем место внизу для кнопок


# --- ФУНКЦИЯ ОТРИСОВКИ ---
def draw_case(idx):
    # Очищаем оси
    ax1.clear()
    ax2.clear()
    ax1.axis("off")
    ax2.axis("off")

    # Получаем путь к текущей папке
    folder_path = subfolders[idx]
    case_name = os.path.basename(folder_path)

    # --- Загрузка Оригинала ---
    possible_images = glob(os.path.join(input_image_dir, f"{case_name}.*"))
    if possible_images:
        original_img = np.array(Image.open(possible_images[0]).convert("L"))
        status_text = "Оригинал найден"
    else:
        # Если нет оригинала, черный квадрат
        temp_mask = glob(os.path.join(folder_path, "*.png"))[0]
        h, w = np.array(Image.open(temp_mask)).shape
        original_img = np.zeros((h, w), dtype=np.uint8)
        status_text = "Оригинал НЕ найден (черный фон)"

    # --- Загрузка Масок ---
    mask_files = sorted(glob(os.path.join(folder_path, "*vertebrae*.png")))

    # --- Рисование ЛЕВОГО окна ---
    ax1.imshow(original_img, cmap="gray")
    ax1.set_title(f"Пациент: {case_name}\n({status_text})")

    # --- Рисование ПРАВОГО окна ---
    ax2.imshow(original_img, cmap="gray")
    ax2.set_title(f"Сегментация ({len(mask_files)} позвонков)")

    if not mask_files:
        ax2.text(0.5, 0.5, "Позвонки не найдены", ha="center", va="center", color="red")
        fig.canvas.draw_idle()
        return

    # Палитра
    cmap = plt.get_cmap("hsv")

    for i, m_file in enumerate(mask_files):
        mask = np.array(Image.open(m_file).convert("L"))
        if np.sum(mask) == 0:
            continue

        name = (
            os.path.basename(m_file)
            .replace(".png", "")
            .replace("vertebrae ", "")
            .upper()
        )
        color = cmap(i / len(mask_files))

        # 1. Заливка
        masked_data = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(
            masked_data, cmap=plt.matplotlib.colors.ListedColormap([color]), alpha=0.4
        )

        # 2. Контур
        ax2.contour(mask, levels=[1], colors=[color], linewidths=1.5)

        # 3. Текст
        coords = np.argwhere(mask > 0)
        y_c, x_c = coords.mean(axis=0)
        ax2.text(
            x_c,
            y_c,
            name,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec=color, alpha=0.7),
        )

    # Обновляем график
    fig.canvas.draw_idle()


# --- КНОПКИ ---
class IndexTracker:
    def __init__(self):
        self.ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind >= len(subfolders):
            self.ind = 0  # Зацикливаем
        draw_case(self.ind)

    def prev(self, event):
        self.ind -= 1
        if self.ind < 0:
            self.ind = len(subfolders) - 1  # Зацикливаем
        draw_case(self.ind)


callback = IndexTracker()

# Создаем кнопки (координаты: x, y, ширина, высота)
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

bnext = Button(axnext, "Вперед >")
bprev = Button(axprev, "< Назад")

bnext.on_clicked(callback.next)
bprev.on_clicked(callback.prev)

# Рисуем первого пациента при старте
draw_case(0)

plt.show()
