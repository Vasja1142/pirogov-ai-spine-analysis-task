from ultralytics import YOLO


def main():
    # УКАЖИТЕ ПУТЬ К ФАЙЛУ last.pt ИЗ ПРЕРВАННОГО ЗАПУСКА
    # Проверьте название папки! (yolo11n_spine_run, run2, run3 и т.д.)
    # Судя по логу, это может быть 'yolo11n_spine_run' или та, что была последней.
    checkpoint_path = "spine_segmentation_project/yolo11n_spine_run/weights/last.pt"

    print(f"🔄 Загружаем чекпоинт: {checkpoint_path}")

    try:
        model = YOLO(checkpoint_path)
    except FileNotFoundError:
        print(
            "❌ Ошибка: Файл last.pt не найден. Проверьте путь к папке spine_segmentation_project!"
        )
        return

    # Возобновляем обучение
    # Параметр resume=True сам подтянет все настройки (эпохи, батч и т.д.) из прошлого запуска
    results = model.train(resume=True)

    print("🎉 Обучение завершено!")


if __name__ == "__main__":
    main()
