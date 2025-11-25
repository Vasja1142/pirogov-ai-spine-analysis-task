"""
Скрипт для удаления данных с малым количеством меток.
Удаляет пары (изображение + метка), если на изображении размечено меньше 5 объектов.
"""

import argparse
from pathlib import Path

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

DATA_DIR = Path("data/02_processed")
MIN_LABELS_COUNT = 6  # Минимальное количество меток (если меньше - удаляем)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def find_corresponding_image(image_dir: Path, label_stem: str) -> Path:
    """Ищет изображение с тем же именем, что и метка."""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_path = image_dir / (label_stem + ext)
        if image_path.exists():
            return image_path
    return None

def process_split(split_path: Path, dry_run: bool) -> int:
    """Обрабатывает одну папку (train или test)."""
    label_dir = split_path / "labels"
    image_dir = split_path / "images"

    if not label_dir.exists() or not image_dir.exists():
        return 0

    print(f"\n--- Проверка папки: {split_path.name} ---")
    
    label_files = list(label_dir.glob("*.txt"))
    deleted_count = 0
    scanty_files = []

    # 1. Поиск файлов
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                # Считаем только непустые строки
                lines = [line.strip() for line in f if line.strip()]
                count = len(lines)
            
            if count < MIN_LABELS_COUNT:
                scanty_files.append((label_file, count))
                
        except Exception as e:
            print(f"Ошибка чтения {label_file.name}: {e}")

    if not scanty_files:
        print("  [OK] Файлов с малым количеством меток не найдено.")
        return 0

    print(f"  Найдено файлов для удаления: {len(scanty_files)}")

    # 2. Удаление
    for label_file, count in scanty_files:
        image_file = find_corresponding_image(image_dir, label_file.stem)
        
        msg = f"    Метки: {count} | Файл: {label_file.name}"
        
        if dry_run:
            print(f"    [DRY RUN] Будет удален (меток: {count}): {label_file.name}")
        else:
            try:
                label_file.unlink()
                print(f"    [DELETED] {label_file.name} (меток: {count})")
                
                if image_file:
                    image_file.unlink()
                
                deleted_count += 1
            except Exception as e:
                print(f"    [ERROR] Не удалось удалить {label_file.name}: {e}")

    return deleted_count

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Удаление изображений с малым числом меток.")
    parser.add_argument('--delete', action='store_true', help="Включить реальное удаление файлов")
    args = parser.parse_args()
    
    dry_run = not args.delete

    print("="*60)
    print(f"Очистка датасета от изображений, где меток < {MIN_LABELS_COUNT}")
    print(f"Папка: {DATA_DIR}")
    print("="*60)

    if dry_run:
        print("РЕЖИМ: Предварительный просмотр (Dry Run).")
        print("Используйте флаг --delete для реального удаления.")
    else:
        print("РЕЖИМ: УДАЛЕНИЕ ФАЙЛОВ!")
        if input("Вы уверены? (yes/no): ").lower() != "yes":
            print("Отмена.")
            return

    total_deleted = 0
    
    # Проверяем train и test (или valid, смотря как называется папка в 02_processed)
    for split_name in ["train", "test", "valid"]:
        split_path = DATA_DIR / split_name
        if split_path.exists():
            total_deleted += process_split(split_path, dry_run)

    print("\n" + "="*60)
    if dry_run:
        print(f"Всего кандидатов на удаление: {total_deleted}")
    else:
        print(f"Всего удалено пар файлов: {total_deleted}")
    print("="*60)

if __name__ == "__main__":
    main()