import os
import torch

from ultralytics import YOLO

def train_model(model_name: str, data_yaml: str, seed: int, epochs: int = 50, imgsz: int = 640):
    """Trains a YOLO segmentation model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(f"{model_name}")
    results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    device=device,
    project=os.path.abspath("runs/segment"),
    name=f"{model_name}_seed_{seed}_training",
    patience=3,            # Early Stopping after 25 epochs without improvement
    workers=4,             # CPU threads (Recommendation: number of physical CPU cores)
    batch=8,               # Further reduced to 8, as VRAM is still limited
    optimizer='AdamW',     # Optimizer
    cache=False,           # Disabled, because cache might fill RAM/VRAM
    seed=seed,             # Dynamic seed value
    overlap_mask=False,
    val = True,
    mask_ratio=1,
    warmup_epochs=3,
    warmup_bias_lr=0.00001,
    lr0=0.01,
    weight_decay=0.0005,
    flipud=0,
    fliplr=0.5,
    mosaic=0,
    amp=True,              # Enables Mixed Precision (halves VRAM usage almost entirely)
    single_cls=False,
    close_mosaic=0,
    save=True,
    save_period=-1,
    deterministic=True     # Recommended for reproducible results
)

    return results
