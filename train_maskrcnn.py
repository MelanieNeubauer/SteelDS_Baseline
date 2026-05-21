import os
import torch
import yaml
from yolo_to_maskrcnn_dataset import get_model_instance_segmentation, YOLOSegDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def train_maskrcnn_model(dataset:str, data_yaml: str, seed: int, epochs: int = 20, imgsz: int = 640, batch_size: int = 8):
    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    num_classes = len(data_dict.get('names', [])) # Background is already in names
    
    model = get_model_instance_segmentation(num_classes, imgsz=imgsz).to(device)
    
    dataset_train = YOLOSegDataset(data_yaml, split='train')
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True
    )
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)
    
    os.makedirs(f"runs{dataset}/maskrcnn_seed_{seed}/weights", exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
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
        
    best_weights = f"runs{dataset}/maskrcnn_seed_{seed}/weights/best.pt"
    torch.save(model.state_dict(), best_weights)
    print(f"Weights saved to {best_weights}")
    return
