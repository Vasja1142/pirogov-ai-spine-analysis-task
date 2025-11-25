import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import cv2
from tqdm import tqdm

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

# Папки, где лежат результаты 09_auto_annotate.py
BASE_DIR = Path("data/01_raw/new_annotated_batch/confident")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

# Имя выходного файла
OUTPUT_XML = Path("data/01_raw/new_annotated_batch/confident/cvat_annotations.xml")

# Имя метки в CVAT (должно совпадать с тем, что ты создал в Task)
LABEL_NAME = "item"

# ============================================================================
# СКРИПТ
# ============================================================================

def create_cvat_xml():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print("Ошибка: Не найдены папки с изображениями или метками.")
        return

    annotations = ET.Element("annotations")
    version = ET.SubElement(annotations, "version")
    version.text = "1.1"

    image_files = sorted(list(IMAGES_DIR.glob("*.*")))
    print(f"Обработка {len(image_files)} изображений...")

    found_labels = 0

    # !!! ВАЖНО: Добавили enumerate, чтобы получать индекс i для ID
    for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
        filename = img_path.name
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        image_elem = ET.SubElement(annotations, "image")
        
        # !!! ИСПРАВЛЕНИЕ: Обязательно добавляем ID
        image_elem.set("id", str(i)) 
        
        image_elem.set("name", filename)
        image_elem.set("width", str(w))
        image_elem.set("height", str(h))

        txt_path = LABELS_DIR / f"{img_path.stem}.txt"
        
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue 
                
                coords_norm = [float(x) for x in parts[1:]]
                
                points_list = []
                for k in range(0, len(coords_norm), 2):
                    x_abs = coords_norm[k] * w
                    y_abs = coords_norm[k+1] * h
                    points_list.append(f"{x_abs:.2f},{y_abs:.2f}")
                
                points_str = ";".join(points_list)

                poly_elem = ET.SubElement(image_elem, "polygon")
                poly_elem.set("label", LABEL_NAME)
                poly_elem.set("points", points_str)
                poly_elem.set("z_order", "0")
                
                found_labels += 1

    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    
    with open(OUTPUT_XML, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print("-" * 40)
    print(f"Готово! XML файл создан: {OUTPUT_XML.absolute()}")
    print("-" * 40)

if __name__ == "__main__":
    create_cvat_xml()