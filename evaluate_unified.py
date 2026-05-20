import os
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from yolo_to_maskrcnn_dataset import get_model_instance_segmentation, YOLOSegDataset

def get_shared_metrics():
    metric_box = MeanAveragePrecision(iou_type="bbox", class_metrics=True, backend="pycocotools")
    metric_seg = MeanAveragePrecision(iou_type="segm", class_metrics=True, backend="pycocotools")
    metric_box_50 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=[0.5], backend="pycocotools")
    metric_seg_50 = MeanAveragePrecision(iou_type="segm", class_metrics=True, iou_thresholds=[0.5], backend="pycocotools")
    return metric_box, metric_seg, metric_box_50, metric_seg_50

def compute_shared_metrics(metric_box, metric_seg, metric_box_50, metric_seg_50, classes):
    """
    Computes all COCO metrics exactly the same way for both models.
    """
    
    res_box = metric_box.compute()
    res_seg = metric_seg.compute()
    res_box_50 = metric_box_50.compute()
    res_seg_50 = metric_seg_50.compute()
    
    out = {}
    out['metrics/mAP50(B)'] = float(res_box_50['map_50']) if res_box_50.get('map_50', torch.tensor([])).numel() > 0 else 0.0
    out['metrics/mAP50-95(B)'] = float(res_box['map']) if res_box.get('map', torch.tensor([])).numel() > 0 else 0.0
    out['metrics/mAP_small(B)'] = float(res_box['map_small']) if res_box.get('map_small', torch.tensor(-1)).numel() > 0 else 0.0
    out['metrics/mAP_medium(B)'] = float(res_box['map_medium']) if res_box.get('map_medium', torch.tensor(-1)).numel() > 0 else 0.0
    out['metrics/mAP_large(B)'] = float(res_box['map_large']) if res_box.get('map_large', torch.tensor(-1)).numel() > 0 else 0.0
    
    out['metrics/mAP50(M)'] = float(res_seg_50['map_50']) if res_seg_50.get('map_50', torch.tensor([])).numel() > 0 else 0.0
    out['metrics/mAP50-95(M)'] = float(res_seg['map']) if res_seg.get('map', torch.tensor([])).numel() > 0 else 0.0
    out['metrics/mAP_small(M)'] = float(res_seg['map_small']) if res_seg.get('map_small', torch.tensor(-1)).numel() > 0 else 0.0
    out['metrics/mAP_medium(M)'] = float(res_seg['map_medium']) if res_seg.get('map_medium', torch.tensor(-1)).numel() > 0 else 0.0
    out['metrics/mAP_large(M)'] = float(res_seg['map_large']) if res_seg.get('map_large', torch.tensor(-1)).numel() > 0 else 0.0
    
    # Class-specific metrics parsing
    for name, res, prefix in [('box', res_box, 'box_mAP50-95'), ('seg', res_seg, 'seg_mAP50-95'),
                              ('box_50', res_box_50, 'box_mAP50'), ('seg_50', res_seg_50, 'seg_mAP50')]:
        if "classes" in res and "map_per_class" in res:
            for i, c in enumerate(res['map_per_class']):
                cls_idx = res['classes'][i].item()
                cls_name = classes.get(cls_idx, f"Class_{cls_idx}")
                out[f"{prefix}_{cls_name}"] = float(c)

    # --- Additional Metrics ---
    out['metrics/recall_mar100(B)'] = float(res_box['mar_100']) if 'mar_100' in res_box and res_box['mar_100'].numel() > 0 else 0.0
    out['metrics/recall_mar100(M)'] = float(res_seg['mar_100']) if 'mar_100' in res_seg and res_seg['mar_100'].numel() > 0 else 0.0
    
    # YOLO Standard Fitness
    out['fitness(B)'] = 0.1 * out['metrics/mAP50(B)'] + 0.9 * out['metrics/mAP50-95(B)']
    out['fitness(M)'] = 0.1 * out['metrics/mAP50(M)'] + 0.9 * out['metrics/mAP50-95(M)']
    out['fitness(Combined)'] = out['fitness(B)'] + out['fitness(M)']

    return out


def evaluate_yolo_model_unified(model_path: str, data_yaml: str, seed: int):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    classes = data_dict.get('names', {0: 'Steel', 1: 'Copper'})
    
    dataset_test = YOLOSegDataset(data_yaml, split='test')
    
    def collate_fn(batch): return tuple(zip(*batch))
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    pred_model = YOLO(model_path)
    
    metric_box, metric_seg, metric_box_50, metric_seg_50 = get_shared_metrics()

    with torch.no_grad():
        for images, targets in data_loader:
            img_paths = [os.path.join(dataset_test.img_dir, dataset_test.img_files[t['image_id'].item()]) for t in targets]
            preds = pred_model(img_paths, verbose=False, conf=0.001)
            
            formatted_targets = []
            formatted_preds = []
            
            for t in targets:
                valid_idx = t["labels"] > 0
                masks_bool = t["masks"][valid_idx].to(device).bool()
                area = masks_bool.sum(dim=(1, 2)).to(device) 
                
                formatted_targets.append({
                    "boxes": t["boxes"][valid_idx].to(device),
                    "labels": (t["labels"][valid_idx].to(device) - 1).clamp(min=0),
                    "masks": masks_bool,
                    "area": area 
                })
                
            for i, p in enumerate(preds):
                H, W = targets[i]['masks'].shape[1:]
                if p.boxes is None or len(p.boxes) == 0:
                    formatted_preds.append({
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                        "masks": torch.zeros((0, H, W), dtype=torch.bool, device=device)
                    })
                    continue
                    
                p_boxes = p.boxes.xyxy.to(device)
                p_scores = p.boxes.conf.to(device)
                p_labels = p.boxes.cls.to(dtype=torch.long, device=device)
                
                if getattr(p, 'masks', None) is not None and getattr(p.masks, 'xy', None) is not None:
                    import numpy as np
                    import cv2
                    p_masks_np = np.zeros((len(p_boxes), H, W), dtype=np.uint8)
                    for mask_idx, poly in enumerate(p.masks.xy):
                        if len(poly) > 0:
                            cv2.fillPoly(p_masks_np[mask_idx], [np.int32(poly)], 1)
                    p_masks = torch.from_numpy(p_masks_np).to(device).bool()
                else:
                    p_masks = torch.zeros((len(p_boxes), H, W), dtype=torch.bool, device=device)
                    
                formatted_preds.append({
                    "boxes": p_boxes,
                    "scores": p_scores,
                    "labels": p_labels,
                    "masks": p_masks
                })
                
            metric_box.update(formatted_preds, formatted_targets)
            metric_seg.update(formatted_preds, formatted_targets)
            metric_box_50.update(formatted_preds, formatted_targets)
            metric_seg_50.update(formatted_preds, formatted_targets)
                
            del preds, formatted_preds, formatted_targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return compute_shared_metrics(metric_box, metric_seg, metric_box_50, metric_seg_50, classes)


def evaluate_maskrcnn_model_unified(model_path: str, data_yaml: str, seed: int, imgsz: int = 640):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    classes = data_dict.get('names', {0: 'Steel', 1: 'Copper'})
    num_classes = len(classes) + 1
    
    model = get_model_instance_segmentation(num_classes, imgsz=imgsz).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataset_test = YOLOSegDataset(data_yaml, split='test')
    
    def collate_fn(batch): return tuple(zip(*batch))
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    metric_box, metric_seg, metric_box_50, metric_seg_50 = get_shared_metrics()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            preds = model(images)
            
            formatted_targets = []
            formatted_preds = []
            
            for t in targets:
                valid_idx = t["labels"] > 0
                masks_bool = t["masks"][valid_idx].to(device).bool()
                area = masks_bool.sum(dim=(1, 2)).to(device)
                
                formatted_targets.append({
                    "boxes": t["boxes"][valid_idx].to(device),
                    "labels": (t["labels"][valid_idx].to(device) - 1).clamp(min=0),
                    "masks": masks_bool,
                    "area": area
                })
                
            for p in preds:
                valid_idx = p["labels"] > 0
                formatted_preds.append({
                    "boxes": p["boxes"][valid_idx],
                    "scores": p["scores"][valid_idx],
                    "labels": (p["labels"][valid_idx] - 1).clamp(min=0),
                    "masks": p["masks"][valid_idx].squeeze(1) > 0.5 
                })
                
            metric_box.update(formatted_preds, formatted_targets)
            metric_seg.update(formatted_preds, formatted_targets)
            metric_box_50.update(formatted_preds, formatted_targets)
            metric_seg_50.update(formatted_preds, formatted_targets)
                
            del preds, images, formatted_preds, formatted_targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return compute_shared_metrics(metric_box, metric_seg, metric_box_50, metric_seg_50, classes)
