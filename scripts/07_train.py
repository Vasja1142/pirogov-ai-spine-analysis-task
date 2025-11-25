from ultralytics import YOLO

# 1. Загружаем стандартную COCO-предобученную модель YOLOv12n
#    Поскольку мы больше не используем YAML, модель будет ожидать 3-канальные изображения.
#    Библиотека сама справится с вашими одноканальными изображениями, дублируя канал.
model = YOLO("data/05_runs/spine_segmentation_v1/weights/best.pt")

# 2. Запускаем обучение с детальными параметрами, включая аугментации
results = model.train(
    task="segment",     # Явно указываем задачу сегментации
    data="data.yaml",
    epochs=15,
    imgsz=640,
    batch=32,
    project="data/05_runs",
    name="spine_segmentation_v2",
    exist_ok=True,
    
    single_cls=True,

    # Аугментации (YOLO сама хорошо крутит полигоны)
    augment=True,
    mosaic=1.0,
    close_mosaic = 5,
    mixup=0.1,
    copy_paste=0.3,     # Очень полезно для сегментации (Copy-Paste augmentation)
    degrees=15,
    translate=0.2,
    scale=0.3,
    fliplr=0.5,
    
    # Параметры сегментации
    overlap_mask=False,  # Маски не могут перекрываться
    mask_ratio=4,       # Downsample ratio для масок (стандарт 4)
    
    patience=50,
    plots=True
)