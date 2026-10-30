import os
import random
import numpy as np
import torch
import yaml
from yolo_to_maskrcnn_dataset import get_model_instance_segmentation, YOLOSegDataset

def collate_fn(batch):
    return tuple(zip(*batch))


def save_checkpoint(path, epoch, model, optimizer):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        # Random-number generator states
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG states LAST
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    if checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return checkpoint["epoch"]


def train_maskrcnn_model(
    dataset: str,
    data_yaml: str,
    seed: int,
    epochs: int = 20,
    imgsz: int = 640,
    batch_size: int = 8,
    resume_from: str = None,
):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    num_classes = len(data_dict.get('names', [])) + 1 # +1 for background

    model = get_model_instance_segmentation(num_classes, imgsz=imgsz, pretrained=False).to(device)

    dataset_train = YOLOSegDataset(data_yaml, split='train')
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)

    weights_dir = f"runs{dataset}/maskrcnn_seed_{seed}/weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Resume from checkpoint, if specified
    start_epoch = 0

    if resume_from is not None:
        start_epoch = load_checkpoint(
            resume_from, model, optimizer, device
        )
        print(f"Resuming from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)

            # Remove empty targets that crash MaskRCNN
            valid_idx = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
            if len(valid_idx) == 0: continue

            images = [images[i] for i in valid_idx]
            targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid_idx]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                print("Warning: NaN Loss in this batch! Skipping...")
                continue

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += losses.item()

        avg_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
        print(f"Mask R-CNN Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # Save complete checkpoint after every epoch
        checkpoint_path = os.path.join(
            weights_dir, f"epoch_{epoch+1:04d}.pth"
        )
        save_checkpoint(
            checkpoint_path, epoch + 1, model, optimizer,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    best_weights = f"{weights_dir}/best.pt"
    torch.save(model.state_dict(), best_weights)
    print(f"Weights saved to {best_weights}")
    return
