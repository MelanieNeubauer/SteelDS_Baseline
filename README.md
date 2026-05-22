# SteelDS: Baseline Evaluation Pipeline

This repository contains the training, evaluation, and visualization scripts for the baseline models (YOLO and Mask R-CNN) on the **SteelDS** dataset.

The codebase allows you to train instance segmentation models, evaluate them on a unified COCO metric scale, and visualize the model predictions on test data.

## Dataset Download

The SteelDS dataset is publicly available on Zenodo. You can download the zipped subset folders (`a1.zip`, `a2.zip`, etc.) from the following DOI link:  
**[10.5281/zenodo.20271102](https://doi.org/10.5281/zenodo.20271102)**

Extract these folders into a directory named `SteelDS` placed alongside this repository.

## Environment Setup

We recommend using an isolated Python environment (e.g., `venv` or `conda`).
The code was tested using **Python 3.13.9** on **Ubuntu 24.04.4 LTS** with an **NVIDIA RTX 4090** GPU (24 GB VRAM).

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The code assumes that the `SteelDS` dataset is located next to this repository folder. Specifically, the data configuration files (`data_a1.yaml`, `data_a2.yaml`, `data_a3.yaml`) point to `../SteelDS/a1`, etc. 

Please ensure your dataset is structured as follows before running the pipeline:

```text
SteelDS/
├── a1/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
├── a2/
│   └── [...]
└── a3/
    └── [...]
```

> **Note:** The labels should be in the YOLO text file format.

## Running the Pipeline

### 1. Training and Evaluation

We provide two main entry points for running the complete pipeline (training + evaluation) across multiple seeds (1337, 7, 42, 99, 123) and subsets (`a1`, `a2`, `a3`). To modify core training hyperparameters—such as batch size, learning rate, or optimizer settings—adjust the configuration directly within the `train.py` source file. Training runs with a batch size of 8 by default and requires 24GB VRAM.

**For YOLO Models (v8, 11, 12, 26):**
```bash
python main_yolo.py
```
This script loops through specified YOLO model architectures, trains them on the defined datasets, evaluates them using a unified COCO metric, and saves the metrics to CSV files in a generated `results_a1` (or respective) directory.


**For Mask R-CNN:**
```bash
python main_maskrcnn.py
```
This script acts similarly for a ResNet50-based Mask R-CNN, generating training loss logs and standardized evaluation metrics.

### 2. Output and Artifacts

The scripts will automatically create folders for artifacts:
- `runs_a1/`, `runs_a2/`, `runs_a3/`: Contains model weights (`best.pt`) and training logs.
- `results_a1/`, `results_a2/`, `results_a3/`: Contains the evaluation summary CSV files with mAP metrics and training plots.

### 3. Visualizing Predictions (Test)

To test the trained models on random test images and visualize the bounding boxes and segmentation masks, use the visualization script:

```bash
python visualize_samples.py
```

This will load the best weights for each model architecture and seed, and export overlaid images into `visualizations_aX/` folders. It generates side-by-side comparisons of bounding box and segmentation mask outputs.

## Customization

- **Changing YOLO architectures**: Edit the `models_to_test` array in `main_yolo.py`.
- **Adjusting Epochs**: Modify the `epochs` parameter passed to `train_model` or `train_maskrcnn_model`.
- **Seeds**: The multi-seed validation seeds can be edited directly in the `main_*.py` scripts.

## License

The code in this repository is licensed under the MIT License. The SteelDS dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
