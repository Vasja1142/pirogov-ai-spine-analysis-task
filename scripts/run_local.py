import os
import cv2
import numpy as np
import torch
import glob
from tkinter import filedialog, Tk
import sys

# --- ИМПОРТЫ SAM 2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Ошибка: Библиотека SAM2 не найдена (pip install sam2).")
    sys.exit(1)

# --- НАСТРОЙКИ MEDSAM-2 ---
# Путь к скачанным весам WangLab
CHECKPOINT_NAME = "data/00_pretrained/MedSAM2_latest.pt"

# WangLab MedSAM-2 обычно использует архитектуру Tiny или Small.
# Попробуем Tiny (t512), так как это дефолт в их репозитории.
# Если будет ошибка mismatch keys, поменяем на 'sam2.1_hiera_s.yaml' (Small) или 'l.yaml' (Large).
# Правильный путь для Tiny модели в SAM 2.1
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ВИЗУАЛИЗАЦИЯ ---
POINT_RADIUS = 6
MASK_ALPHA = 0.4
COLOR_POS = (0, 255, 0)
COLOR_NEG = (255, 0, 0)

class MedSAM2Annotator:
    def __init__(self):
        self.images_path = []
        self.current_img_idx = 0
        self.class_id = 0 
        
        self.orig_image = None
        self.masks_storage = [] 
        self.active_points = [] 
        self.active_labels = [] 
        self.current_mask_preview = None 

        print(f"Устройство: {DEVICE}")
        
        # Поиск весов
        if not os.path.exists(CHECKPOINT_NAME):
            # Ищем относительно скрипта
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            alt_path = os.path.join(project_root, CHECKPOINT_NAME)
            if os.path.exists(alt_path):
                self.checkpoint_path = alt_path
            else:
                print(f"ОШИБКА: Не найден файл {CHECKPOINT_NAME}")
                print("Скачайте его с HuggingFace: wanglab/MedSAM2")
                sys.exit(1)
        else:
            self.checkpoint_path = CHECKPOINT_NAME

        print(f"Загрузка MedSAM-2 (Config: {MODEL_CFG})...")
        try:
            # Загружаем модель
            self.model = build_sam2(MODEL_CFG, self.checkpoint_path, device=DEVICE)
            self.predictor = SAM2ImagePredictor(self.model)
            print(">>> MedSAM-2 успешно загружен!")
        except Exception as e:
            print(f"\nОШИБКА ЗАГРУЗКИ: {e}")
            print("Возможно, не совпадает конфиг (.yaml). Попробуйте в скрипте поменять MODEL_CFG на 'sam2.1_hiera_s.yaml' или 'l.yaml'")
            sys.exit(1)

    def select_folder(self):
        root = Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory(title="Выберите папку с рентгенами")
        root.destroy()
        if not folder_selected: return False
        
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.dcm']
        self.images_path = []
        for ext in exts:
            self.images_path.extend(glob.glob(os.path.join(folder_selected, ext)))
        self.images_path.sort()
        
        if not self.images_path:
            print("Файлы не найдены.")
            return False

        parent_dir = os.path.dirname(folder_selected)
        dir_name = os.path.basename(folder_selected)
        if dir_name == 'images':
            self.labels_dir = os.path.join(parent_dir, 'labels')
        else:
            self.labels_dir = os.path.join(folder_selected, 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)
        return True

    def preprocess_image(self, img):
        # CLAHE для контраста (MedSAM любит четкие границы)
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        except:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_current_image(self):
        path = self.images_path[self.current_img_idx]
        self.orig_image = cv2.imread(path)
        if self.orig_image is None: return False
        
        # Препроцессинг
        img_input = self.preprocess_image(self.orig_image)
        self.predictor.set_image(img_input)
        
        self.masks_storage = []
        self.active_points = []
        self.active_labels = []
        self.current_mask_preview = None
        return True

    def save_yolo_format(self):
        if not self.masks_storage: return
        img_h, img_w = self.orig_image.shape[:2]
        img_name = os.path.basename(self.images_path[self.current_img_idx])
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        save_path = os.path.join(self.labels_dir, txt_name)

        lines = []
        for mask in self.masks_storage:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                epsilon = 0.001 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                points = []
                for p in approx:
                    x, y = p[0]
                    nx = min(max(x / img_w, 0), 1)
                    ny = min(max(y / img_h, 0), 1)
                    points.append(f"{nx:.6f} {ny:.6f}")
                if len(points) > 2:
                    line = f"{self.class_id} " + " ".join(points)
                    lines.append(line)

        with open(save_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Сохранено: {txt_name}")

    def predict_active(self):
        if not self.active_points:
            self.current_mask_preview = None
            return

        points_np = np.array(self.active_points)
        labels_np = np.array(self.active_labels)

        # SAM 2 API
        masks, scores, logits = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=False 
        )
        self.current_mask_preview = masks[0] > 0.5

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.active_points.append([x, y])
                self.active_labels.append(0)
            else:
                self.active_points.append([x, y])
                self.active_labels.append(1)
            self.predict_active()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.commit_mask()

    def commit_mask(self):
        if self.current_mask_preview is not None:
            self.masks_storage.append(self.current_mask_preview)
            self.active_points = []
            self.active_labels = []
            self.current_mask_preview = None
            print(f"Объект добавлен ({len(self.masks_storage)})")

    def draw_overlay(self):
        display_img = self.orig_image.copy()
        mask_layer = np.zeros_like(display_img)
        
        for m in self.masks_storage:
            mask_layer[m] = (200, 100, 0)
        if self.current_mask_preview is not None:
            mask_layer[self.current_mask_preview] = (0, 255, 100)

        display_img = cv2.addWeighted(display_img, 1.0, mask_layer, MASK_ALPHA, 0)
        
        for pt, lbl in zip(self.active_points, self.active_labels):
            color = COLOR_POS if lbl == 1 else COLOR_NEG
            cv2.circle(display_img, tuple(pt), POINT_RADIUS, color, -1)
            cv2.circle(display_img, tuple(pt), POINT_RADIUS, (255,255,255), 1)

        info_txt = f"File: {self.current_img_idx+1}/{len(self.images_path)} | Count: {len(self.masks_storage)}"
        cv2.putText(display_img, info_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return display_img

    def run(self):
        if not self.select_folder(): return
        if not self.load_current_image(): return

        win_name = "MedSAM-2 Annotator (Nov 2025)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        while True:
            vis = self.draw_overlay()
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('c'): self.commit_mask()
            elif key == ord('d'):
                if self.active_points:
                    self.active_points.pop()
                    self.active_labels.pop()
                    self.predict_active()
            elif key == ord('r'):
                self.active_points = []
                self.active_labels = []
                self.current_mask_preview = None
            elif key == ord('s') or key == 32:
                if self.current_mask_preview is not None: self.commit_mask()
                self.save_yolo_format()
                self.current_img_idx += 1
                if self.current_img_idx >= len(self.images_path): break
                self.load_current_image()
            elif key == ord('q'): break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MedSAM2Annotator()
    app.run()