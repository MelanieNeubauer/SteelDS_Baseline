import os
import cv2
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class YOLOSegDataset(Dataset):
    def __init__(self, data_yaml, split='train'):
        with open(data_yaml, 'r') as f:
            self.data_dict = yaml.safe_load(f)
            
        base_path = self.data_dict.get('path', '')
        split_path = self.data_dict[split]
        
        if isinstance(split_path, str):
            if os.path.isabs(split_path):
                self.img_dir = split_path
            else:
                self.img_dir = os.path.join(base_path, split_path)
        else:
            self.img_dir = os.path.join(base_path, split_path[0])
            
        self.lbl_dir = self.img_dir.replace('images', 'labels')
        self.img_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        labels = []
        masks = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_idx = int(parts[0])
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)
                
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    # YOLO labels now match Mask R-CNN expected indices (0=Bg, 1=Steel, 2=Copper)
                    labels.append(cls_idx) 
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.int32(coords)], 1)
                    masks.append(mask)

        img_tensor = TF.to_tensor(img)
        
        target = {}
        if len(boxes) > 0:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['masks'] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, h, w), dtype=torch.uint8)
            
        target['image_id'] = torch.tensor([idx])
        return img_tensor, target
        
def get_model_instance_segmentation(num_classes, imgsz=640):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # Initialize model with default pre-trained weights
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights, min_size=imgsz, max_size=imgsz, box_score_thresh=0.001)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
