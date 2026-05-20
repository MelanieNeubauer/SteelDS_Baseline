import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(csv_path: str, save_path: str = "training_plot.png"):
    """Reads the training results.csv and generates metric plots."""
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(12, 5))

    # Plot mAP
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP@50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP@50-95')
    plt.title('Mean Average Precision (mAP)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)

    # Plot Losses
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/seg_loss'], label='Train Seg Loss')
    plt.plot(df['epoch'], df['val/seg_loss'], label='Val Seg Loss')
    plt.title('Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")