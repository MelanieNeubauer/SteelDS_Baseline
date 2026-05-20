import os
# Allows PyTorch to manage VRAM more efficiently (must be placed before the first 'import torch'!):
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import pandas as pd
import gc
import torch
from train import train_model
from evaluate_unified import evaluate_yolo_model_unified
from plot import plot_training_results
from ultralytics import settings

def main_yolo(dataset='_a1'):

    data_yaml = f"data{dataset}.yaml"

    # Validation across multiple models to ensure data quality
    models_to_test = ["yolov8n-seg.yaml", "yolo26n-seg.yaml", "yolo11n-seg.yaml","yolo12n-seg.yaml"]
    seeds = [1337, 7, 42, 99, 123]
    
    os.makedirs(f"results{dataset}", exist_ok=True)
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"--- Starting validation pipeline for {model_name} ---")
        print(f"{'='*50}")
        
        all_eval_metrics = []
        
        for seed in seeds:
            print(f"\n>> Starting seed {seed} for {model_name} <<")
            
            # 1. Training
            train_model(model_name=model_name, data_yaml=data_yaml, seed=seed, epochs=20)
            
            # 2. Evaluation
            run_dir = os.path.abspath(f"runs{dataset}/segment/{model_name}_seed_{seed}_training")
            best_weights = os.path.join(run_dir, "weights", "best.pt")
            eval_metrics = evaluate_yolo_model_unified(model_path=best_weights, data_yaml=data_yaml, seed=seed)
            
            # Save metrics for later CSV export
            # eval_metrics is a dictionary with YOLO values (metrics.results_dict)
            metrics_row = {"seed": seed}
            metrics_row.update(eval_metrics)
            all_eval_metrics.append(metrics_row)
            
            # Save all metrics bundled in ONE single file (updated after each seed)
            eval_csv_path = f"results{dataset}/{model_name}_evaluation_metrics_5_seeds.csv"
            pd.DataFrame(all_eval_metrics).to_csv(eval_csv_path, index=False)
            
            # 3. Copy results
            results_csv = os.path.join(run_dir, "results{dataset}.csv")
            destination_csv = f"results{dataset}/{model_name}_seed_{seed}_train_results.csv"
            
            if os.path.exists(results_csv):
                shutil.copy(results_csv, destination_csv)
                print(f"📄 Training data saved as CSV for seed {seed} at: {destination_csv}")

            # Generate plot
            plot_training_results(
                csv_path=results_csv, 
                save_path=f"results{dataset}/{model_name}_seed_{seed}_training_plot.png"
            )
            
            # Clean up RAM and VRAM (if GPU is used)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        print(f"\n--- All seeds for {model_name} completed ---")
        
        # Save evaluation summary as CSV
        eval_df = pd.DataFrame(all_eval_metrics)
        eval_csv_path = f"results{dataset}/{model_name}_evaluation_metrics_5_seeds.csv"
        eval_df.to_csv(eval_csv_path, index=False)
        print(f"📊 Evaluation data across all 5 seeds for {model_name} saved in: {eval_csv_path}")

#'''
if __name__ == "__main__":
    main_yolo()
    #'''