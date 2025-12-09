import zipfile
import numpy as np
import os
from io import BytesIO

# Путь к вашему огромному архиву
zip_path = "data/01_raw/PAX-RayPlusPlus/labels.zip"
# Куда сохранять сжатые файлы
output_dir = "data/01_raw/PAX-RayPlusPlus/labels"

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    # Получаем список файлов
    file_list = [f for f in z.namelist() if f.endswith(".npy")]

    print(f"Найдено файлов: {len(file_list)}")

    for i, filename in enumerate(file_list):
        # Читаем файл прямо из архива в память (не распаковывая на диск)
        with z.open(filename) as f:
            # BytesIO нужен, так как np.load хочет файл-подобный объект
            file_content = BytesIO(f.read())
            data = np.load(file_content)

        # Сохраняем в сжатом формате .npz
        # Имя файла будет name.npz вместо name.npy
        base_name = os.path.basename(filename).replace(".npy", "")
        save_path = os.path.join(output_dir, base_name)

        # Эта команда сожмет данные в десятки/сотни раз
        np.savez_compressed(save_path, data=data)

        if i % 100 == 0:
            print(f"Обработано {i} файлов...")

print("Готово! Теперь папка весит адекватно.")
