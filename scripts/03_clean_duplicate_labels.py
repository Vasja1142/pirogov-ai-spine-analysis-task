"""
Скрипт для поиска и удаления файлов с дубликатами меток.

Функциональность:
- Находит файлы аннотаций с дублирующимися метками классов
- Удаляет найденные файлы аннотаций и соответствующие изображения
- Поддерживает режим "сухого запуска" для предварительного анализа
- Работает с train, valid и test splits
"""

import argparse
from collections import Counter
from pathlib import Path


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

DATA_DIR = Path("data/02_processed")


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def find_corresponding_image(image_dir: Path, label_stem: str) -> Path:
    """
    Находит файл изображения, соответствующий файлу метки.
    
    Args:
        image_dir: Директория с изображениями
        label_stem: Имя файла метки без расширения
    
    Returns:
        Path к изображению или None, если не найдено
    """
    for ext in ['.jpg', '.jpeg', '.png']:
        image_path = image_dir / (label_stem + ext)
        if image_path.exists():
            return image_path
    return None


def process_split(split_path: Path, dry_run: bool) -> int:
    """
    Находит и удаляет файлы с дубликатами меток в одном split.
    
    Args:
        split_path: Путь к split (train/valid/test)
        dry_run: Если True, только показывает что будет удалено
    
    Returns:
        Количество удаленных пар файлов
    """
    label_dir = split_path / "labels"
    image_dir = split_path / "images"

    if not label_dir.exists() or not image_dir.exists():
        print(
            f"  [!] В сплите '{split_path.name}' не найдены папки "
            f"'labels' или 'images'. Пропуск."
        )
        return 0

    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        return 0

    print(
        f"\n--- Проверка сплита: '{split_path.name}' "
        f"({len(label_files)} файлов) ---"
    )
    
    files_to_delete = []

    # Поиск файлов с дубликатами
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                class_ids = [
                    int(line.strip().split()[0])
                    for line in f
                    if line.strip()
                ]
            
            # Подсчет количества каждой метки
            counts = Counter(class_ids)
            
            # Проверка на дубликаты
            if any(count >= 2 for count in counts.values()):
                files_to_delete.append(label_file)

        except (IndexError, ValueError) as e:
            print(
                f"  [!] Ошибка чтения файла {label_file.name}: {e}. "
                f"Файл может быть поврежден."
            )
        except Exception as e:
            print(
                f"  [!] Неизвестная ошибка при обработке "
                f"{label_file.name}: {e}"
            )

    if not files_to_delete:
        print("  [OK] Дубликатов меток не найдено.")
        return 0
    
    print(f"  [!] Найдено {len(files_to_delete)} файлов с дубликатами меток:")
    
    deleted_count = 0
    
    # Удаление или отображение файлов для удаления
    for label_file in files_to_delete:
        image_file = find_corresponding_image(image_dir, label_file.stem)
        
        if dry_run:
            print(f"    (Dry Run) Будет удален: {label_file.name}")
            if image_file:
                print(f"    (Dry Run) Будет удален: {image_file.name}")
        else:
            try:
                print(f"  [DELETING] Удаление {label_file.name}...")
                label_file.unlink()
                
                if image_file:
                    print(f"  [DELETING] Удаление {image_file.name}...")
                    image_file.unlink()
                
                deleted_count += 1
                
            except Exception as e:
                print(
                    f"  [!!!] ОШИБКА при удалении {label_file.name}: {e}"
                )
    
    if dry_run:
        print(
            f"\n  [INFO] 'Сухой запуск' завершен. "
            f"Для реального удаления запустите с флагом --delete."
        )
    
    return deleted_count


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция для запуска очистки."""
    
    parser = argparse.ArgumentParser(
        description="Скрипт для поиска и удаления изображений с дублирующимися метками.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help=(
            "Включить реальное удаление файлов.\n"
            "По умолчанию скрипт работает в режиме 'сухого запуска' (dry run)."
        )
    )
    
    args = parser.parse_args()
    dry_run = not args.delete

    # Заголовок
    print("="*60)
    print("      Скрипт для очистки дубликатов меток")
    print("="*60)
    print(f"[*] Целевая папка: {DATA_DIR}")

    if dry_run:
        print("[*] РЕЖИМ: Сухой запуск (Dry Run). Файлы не будут удалены.")
    else:
        print("\n" + "!"*60)
        print("      ВНИМАНИЕ: РЕЖИМ РЕАЛЬНОГО УДАЛЕНИЯ АКТИВИРОВАН!")
        print("      Эта операция необратима. Убедитесь, что у вас есть бэкап.")
        print("!"*60 + "\n")
        
        confirm = input("==> Для подтверждения удаления введите 'yes': ")
        if confirm.lower() != 'yes':
            print("[INFO] Удаление отменено пользователем.")
            return

    # Обработка каждого split
    total_deleted = 0
    for split_name in ["train", "valid", "test"]:
        split_path = DATA_DIR / split_name
        if split_path.exists():
            deleted_in_split = process_split(split_path, dry_run)
            total_deleted += deleted_in_split
    
    # Итоговая статистика
    print("\n" + "="*60)
    if dry_run:
        print(
            f"      Анализ завершен. "
            f"Всего найдено потенциальных дубликатов: {total_deleted}"
        )
    else:
        print(
            f"      Очистка завершена. "
            f"Всего удалено пар файлов: {total_deleted}"
        )
    print("="*60)


if __name__ == "__main__":
    main()