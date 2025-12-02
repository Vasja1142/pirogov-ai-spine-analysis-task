"""
Скрипт для создания `dataset.yaml` файла для YOLO.

Этот скрипт автоматически генерирует YAML файл, который содержит пути
к набору данных и имена классов, необходимые для обучения моделей YOLO.
Он проверяет наличие и непустоту директорий `train`, `test` и/или `valid`.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Union

# ============================================================================
# ⚙️ ФУНКЦИИ
# ============================================================================

def parse_class_names(names_arg: str) -> Dict[int, str]:
    """
    Парсит строку с именами классов в словарь.

    Примеры входных строк:
    - "0:vertebra,1:disk"
    - "vertebra,disk"

    Args:
        names_arg: Строка с именами классов.

    Returns:
        Словарь, где ключ - это ID класса, а значение - имя класса.
    """
    class_names: Dict[int, str] = {}
    try:
        if ":" in names_arg:
            # Формат "0:vertebra,1:disk"
            for pair in names_arg.split(','):
                key, value = pair.split(':')
                class_names[int(key.strip())] = value.strip()
        else:
            # Формат "vertebra,disk"
            for i, name in enumerate(names_arg.split(',')):
                class_names[i] = name.strip()
    except ValueError as e:
        print(f"❌ Ошибка парсинга имен классов: {e}. Проверьте формат.")
        return {}
    return class_names

def find_data_split(
    dataset_dir: Path, primary_name: str, fallback_name: str
) -> Union[Path, None]:
    """
    Ищет директорию набора данных, проверяя основное и запасное имя.

    Args:
        dataset_dir: Корневая директория набора данных.
        primary_name: Основное имя для поиска (например, 'test').
        fallback_name: Запасное имя (например, 'valid').

    Returns:
        Путь к найденной директории или None.
    """
    primary_path = dataset_dir / "images" / primary_name
    if primary_path.is_dir() and any(primary_path.iterdir()):
        return primary_path

    fallback_path = dataset_dir / "images" / fallback_name
    if fallback_path.is_dir() and any(fallback_path.iterdir()):
        print(f"  [Предупреждение] Директория '{primary_name}' не найдена или пуста, используется '{fallback_name}'.")
        return fallback_path

    return None

def create_yaml_config(
    dataset_dir: Path, class_names: Dict[int, str]
) -> bool:
    """
    Проверяет структуру данных и создает `dataset.yaml`.

    Args:
        dataset_dir: Абсолютный путь к директории с набором данных.
        class_names: Словарь с именами классов.

    Returns:
        True, если файл успешно создан, иначе False.
    """
    # 1. Проверка директорий
    train_dir = dataset_dir / "images" / "train"
    val_dir = find_data_split(dataset_dir, 'test', 'valid')

    if not train_dir.is_dir() or not any(train_dir.iterdir()):
        print(f"❌ Ошибка: Директория '{train_dir}' не найдена или пуста.")
        return False

    if not val_dir:
        print(f"❌ Ошибка: Не найдены или пусты директории для валидации ('images/test' или 'images/valid').")
        return False

    print("✅ Структура директорий корректна.")
    print(f"  - Train: {len(list(train_dir.glob('*.*')))} изображений")
    print(f"  - Val:   {len(list(val_dir.glob('*.*')))} изображений")

    # 2. Формирование данных для YAML
    yaml_data = {
        'path': dataset_dir.as_posix(),
        'train': 'images/train',
        'val': f"images/{val_dir.name}",
        'test': f"images/{val_dir.name}",
        'names': class_names,
    }

    # 3. Сохранение файла
    yaml_path = dataset_dir / "dataset.yaml"
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    except IOError as e:
        print(f"❌ Ошибка записи файла {yaml_path}: {e}")
        return False

    print(f"\n✅ Файл конфигурации успешно создан: {yaml_path}")
    print("Содержимое:")
    print("-" * 25)
    print(yaml.dump(yaml_data, sort_keys=False, allow_unicode=True))
    print("-" * 25)
    print(f"Используйте его для обучения: data='{yaml_path.as_posix()}'")
    return True

# ============================================================================
# 🚀 ЗАПУСК
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Создание `dataset.yaml` для YOLO.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/04_normalized"),
        help="Путь к корневой директории обработанного набора данных."
    )
    parser.add_argument(
        "--names",
        type=str,
        default="0:vertebra",
        help="Имена классов в формате '0:name1,1:name2' или 'name1,name2'."
    )
    args = parser.parse_args()

    dataset_path = args.path.resolve()
    print(f"🚀 Генерация YAML для директории: {dataset_path}")

    if not dataset_path.is_dir():
        print(f"❌ Ошибка: Указанная директория не существует: {dataset_path}")
        return

    class_names = parse_class_names(args.names)
    if not class_names:
        print("❌ Операция прервана из-за ошибки в именах классов.")
        return

    create_yaml_config(dataset_path, class_names)

if __name__ == "__main__":
    main()