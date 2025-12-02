"""
Скрипт для обучения модели YOLO11 Segmentation.

Позволяет настроить параметры обучения через аргументы командной строки,
такие как путь к файлу конфигурации данных, количество эпох, размер батча,
имя базовой модели и название проекта для сохранения результатов.

Пример использования:
    python scripts/07_train.py --data data/04_normalized/dataset.yaml --epochs 50 --batch 16
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Обучение модели YOLO11 Segmentation.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/04_normalized/dataset.yaml"),
        help="Путь к файлу dataset.yaml. По умолчанию: data/04_normalized/dataset.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Количество эпох обучения. По умолчанию: 30.",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Размер батча. По умолчанию: 16."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo11n-seg.pt",
        help="Имя базовой модели YOLO (например, yolo11n-seg.pt, yolo11s-seg.pt).",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="data/05_runs",
        help="Папка для сохранения результатов обучения. По умолчанию: data/05_runs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="spine_segmentation_v11",
        help="Имя текущего эксперимента. По умолчанию: spine_segmentation_v11.",
    )

    args = parser.parse_args()

    data_yaml_path = args.data.resolve()

    if not data_yaml_path.exists():
        print(f"❌ Ошибка: Файл конфигурации данных не найден: {data_yaml_path}")
        print(
            "Пожалуйста, убедитесь, что вы запустили scripts/06_create_yaml.py и указали верный путь."
        )
        return

    # 1. Загружаем модель
    print(f"[*] Загрузка базовой модели: {args.model_name}")
    try:
        model = YOLO(args.model_name)
    except Exception as e:
        print(
            f"❌ Ошибка при загрузке модели. Убедитесь, что установлен ultralytics>=8.3.0 для поддержки YOLO11."
        )
        raise e

    # 2. Запускаем обучение
    print(f"🚀 Запуск обучения с конфигом: {data_yaml_path}")
    print(f"⚙️ Параметры: Эпохи={args.epochs}, Батч={args.batch}")

    results = model.train(
        task="segment",
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        project=args.project_name,
        name=args.experiment_name,
        exist_ok=True,
        single_cls=True,
        augment=True,
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.2,
        copy_paste=0.3,
        degrees=20,
        translate=0.2,
        scale=0.3,
        fliplr=0.5,
        patience=50,
        plots=True,
        workers=4,
        perspective=0.0008,
        shear=0.2,
    )

    print("✅ Обучение завершено!")
    if results.save_dir:
        print(f"Результаты сохранены в: {Path(results.save_dir).resolve()}")


if __name__ == "__main__":
    main()
