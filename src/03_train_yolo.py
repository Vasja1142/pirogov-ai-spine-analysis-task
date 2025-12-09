from ultralytics import YOLO


def main():
    # 1. Загружаем модель
    # yolo11n-seg.pt - Nano (быстрая). Если есть мощная GPU, можно попробовать yolo11s-seg.pt (Small) или m (Medium)
    model = YOLO("yolo11n-seg.pt")

    # 2. ВАЖНО: Путь к конфигу УЛУЧШЕННОГО датасета
    data_config = "data/03_enhanced/dataset.yaml"

    # 3. Запускаем обучение
    results = model.train(
        data=data_config,
        # --- Основные параметры ---
        epochs=25,
        close_mosaic=5,
        exist_ok=True,
        imgsz=640,  # Размер входного изображения
        batch=48,
        device=0,  # Ваша GPU
        # --- Параметры сохранения ---
        project="data/05_run",
        name="run_data_enhanced_only_clahe",  # Назовем так, чтобы помнить, что это на улучшенных данных
        # --- Важно для медицинских снимков ---
        # Отключаем HSV аугментацию (изменение цвета), так как у нас ЧБ рентген
        hsv_h=0.0,  # Hue (Оттенок) - отключаем
        hsv_s=0.0,  # Saturation (Насыщенность) - отключаем
        hsv_v=0.4,  # Value (Яркость) - оставляем, полезно
    )

    print("Обучение завершено!")


if __name__ == "__main__":
    main()
