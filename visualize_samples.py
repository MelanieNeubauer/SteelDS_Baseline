import os
import shutil
import random
import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as TF
from yolo_to_maskrcnn_dataset import get_model_instance_segmentation, YOLOSegDataset

COLORS_RGB = {
    'Steel': (0, 0, 255),
    'Copper': (0, 255, 0)
}
COLORS_BGR = {
    'Steel': (255, 0, 0),
    'Copper': (0, 255, 0)
}
ALPHA = 0.3

def get_random_test_images(data_yaml_path, num_samples=3):
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
        
    base_path = data_dict.get('path', '')
    split_path = data_dict['test']
    
    if isinstance(split_path, str):
        if os.path.isabs(split_path):
            img_dir = split_path
        else:
            img_dir = os.path.join(base_path, split_path)
    else:
        img_dir = os.path.join(base_path, split_path[0])
        
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_imgs = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    
    return [os.path.join(img_dir, img) for img in selected_imgs]

def draw_and_save(img_tensor, boxes, masks, string_labels, scores, output_dir, model_name, img_name):
    img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()
    
    if scores is not None:
        text_labels = [f"{lbl} {score * 100:.1f}%" for lbl, score in zip(string_labels, scores)]
    else:
        text_labels = string_labels
        
    box_colors_str = [f"rgb({COLORS_RGB.get(lbl, (255,255,255))[0]}, {COLORS_RGB.get(lbl, (255,255,255))[1]}, {COLORS_RGB.get(lbl, (255,255,255))[2]})" for lbl in string_labels]
    
    # 1. Boxes Image
    img_boxes = img_uint8.clone()
    if len(boxes) > 0:
        # We don't pass labels here because torchvision ignores font_size without a custom .ttf font.
        img_boxes = draw_bounding_boxes(img_boxes, torch.as_tensor(boxes).cpu(), colors=box_colors_str, width=10)
    final_img_boxes = cv2.cvtColor(img_boxes.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    
    # 2. Segmentation Image
    img_masks = img_uint8.clone()
    if len(boxes) > 0 and masks is not None and len(masks) > 0:
        for idx in range(len(masks)):
            cls_name = string_labels[idx]
            color = COLORS_RGB.get(cls_name, (255, 255, 255))
            color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
            mask_tensor = torch.as_tensor(masks[idx]).unsqueeze(0).bool().cpu()
            
            img_masks = draw_segmentation_masks(img_masks, mask_tensor, alpha=ALPHA, colors=[color_str])
            
    final_img_masks = cv2.cvtColor(img_masks.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    
    # Add text to both images using cv2
    if len(boxes) > 0:
        drawn_positions = []
        for idx in range(len(boxes)):
            box = boxes[idx]
            text = text_labels[idx]
            cls_name = string_labels[idx]
            color_bgr = COLORS_BGR.get(cls_name, (0, 0, 0)) # fallback
            
            x, y = int(box[0]), int(box[1])
            y = max(y, 45) # ensure it doesn't go off top screen
            
            # Smart collision detection to prevent overlapping text for different objects
            while any(abs(x - px) < 250 and abs(y - py) < 60 for (px, py) in drawn_positions):
                y += 60 # Push the text down if it overlaps
                
            drawn_positions.append((x, y))
            
            # Using fontScale=2.0 and thickness=4
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            
            # Draw white background rectangle for maximum readability
            cv2.rectangle(final_img_boxes, (x, y - th - 5), (x + tw, y + baseline + 5), (255, 255, 255), -1)
            cv2.rectangle(final_img_masks, (x, y - th - 5), (x + tw, y + baseline + 5), (255, 255, 255), -1)
            
            # Draw bold text
            cv2.putText(final_img_boxes, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_bgr, 4, cv2.LINE_AA)
            cv2.putText(final_img_masks, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_bgr, 4, cv2.LINE_AA)
            
    cv2.imwrite(os.path.join(output_dir, 'boxes', f"{model_name}_boxes_{img_name}"), final_img_boxes)
    cv2.imwrite(os.path.join(output_dir, 'segmentation', f"{model_name}_masks_{img_name}"), final_img_masks)


def visualize_yolo(model_path, data_yaml_path, img_paths, output_dir, model_name):
    if not os.path.exists(model_path):
        return
        
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    classes = data_dict.get('names', {0: 'Background', 1: 'Steel', 2: 'Copper'})
        
    model = YOLO(model_path)
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb)
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        
        res = model(img_path, verbose=False, conf=0.25)
        
        boxes = []
        masks = []
        string_labels = []
        scores = []
        
        if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
            b = res[0].boxes
            boxes = b.xyxy.cpu().numpy()
            scores = b.conf.cpu().numpy()
            class_indices = b.cls.cpu().numpy().astype(int)
            string_labels = [classes.get(idx, 'Unknown') for idx in class_indices]
            
            if getattr(res[0], 'masks', None) is not None and getattr(res[0].masks, 'xy', None) is not None:
                for poly in res[0].masks.xy:
                    mask_np = np.zeros((H, W), dtype=np.uint8)
                    if len(poly) > 0:
                        cv2.fillPoly(mask_np, [np.int32(poly)], 1)
                    masks.append(mask_np)
            else:
                masks = [np.zeros((H, W), dtype=np.uint8) for _ in boxes]
                
        draw_and_save(img_tensor, boxes, masks, string_labels, scores if len(scores) > 0 else None, output_dir, model_name, img_name)

def visualize_maskrcnn(model_path, data_yaml_path, img_paths, output_dir, model_name="maskrcnn"):
    if not os.path.exists(model_path):
        return
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    classes = data_dict.get('names', {0: 'Background', 1: 'Steel', 2: 'Copper'})
    num_classes = len(classes)
    
    model = get_model_instance_segmentation(num_classes, imgsz=640).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).to(device)
        
        with torch.no_grad():
            preds = model([img_tensor])[0]
            
        keep = preds['scores'] > 0.5
        boxes = preds['boxes'][keep].cpu().numpy()
        masks = (preds['masks'][keep].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        labels = preds['labels'][keep].cpu().numpy()
        scores = preds['scores'][keep].cpu().numpy()
        
        string_labels = [classes.get(l, 'Unknown') for l in labels]
        
        draw_and_save(img_tensor.cpu(), boxes, masks, string_labels, scores if len(scores) > 0 else None, output_dir, model_name, img_name)

def visualize_ground_truth(data_yaml_path, img_paths, output_dir):
    dataset = YOLOSegDataset(data_yaml_path, split='test')
    
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    classes = data_dict.get('names', {0: 'Background', 1: 'Steel', 2: 'Copper'})
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        try:
            idx = dataset.img_files.index(img_name)
        except ValueError:
            continue
            
        img_tensor, target = dataset[idx]
        
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        masks = target['masks'].numpy()
        
        string_labels = [classes.get(l, 'Unknown') for l in labels]
        
        draw_and_save(img_tensor, boxes, masks, string_labels, None, output_dir, "ground_truth", img_name)

def main():
    print("Welcome to model visualization!")
    
    datasets = ["_a1", "_a2", "_a3"]
    seed = 1337
    num_samples = 3
    
    for dataset in datasets:
        data_yaml = f"data{dataset}.yaml"
        if not os.path.exists(data_yaml):
            continue
            
        print(f"\nProcessing dataset {dataset}...")
        
        # Create output folders
        output_dir = f"visualizations{dataset}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'boxes'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'segmentation'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ground_truth'), exist_ok=True)
        
        # Get random images
        img_paths = get_random_test_images(data_yaml, num_samples=num_samples)
        if not img_paths:
            print("No test images found!")
            continue
            
        # Draw ground truth
        visualize_ground_truth(data_yaml, img_paths, output_dir)
            
        yolo_models = ["yolov8n-seg.yaml", "yolo11n-seg.yaml", "yolo12n-seg.yaml", "yolo26n-seg.yaml"]
        
        for model_name in yolo_models:
            run_dir = os.path.abspath(f"runs{dataset}/segment/{model_name}_seed_{seed}_training")
            best_weights = os.path.join(run_dir, "weights", "best.pt")
            try:
                visualize_yolo(best_weights, data_yaml, img_paths, output_dir, model_name)
            except Exception as e:
                print(f"Error with YOLO {model_name}: {e}")
            
        # Mask R-CNN
        maskrcnn_weights = f"runs{dataset}/maskrcnn_seed_{seed}/weights/best.pt"
        try:
            visualize_maskrcnn(maskrcnn_weights, data_yaml, img_paths, output_dir, "maskrcnn-resnet50")
        except Exception as e:
            print(f"Error with Mask R-CNN: {e}")
        
    print("\nVisualizations completed! See the 'visualizations_aX' folders.")

if __name__ == "__main__":
    main()
