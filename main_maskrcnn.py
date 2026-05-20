import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import pandas as pd
import gc
import torch
from train_maskrcnn import train_maskrcnn_model
from evaluate_unified import evaluate_maskrcnn_model_unified

def main_maskrcnn(dataset='_a1'):
    data_yaml = f"data{dataset}.yaml"

    models_to_test = ["maskrcnn-resnet50"]
    seeds = [1337, 7, 42, 99, 123]
    
    os.makedirs(f"results{dataset}", exist_ok=True)
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"--- Starting validation pipeline for {model_name} ---")
        
        all_eval_metrics = []
        
        for seed in seeds:
            print(f"\n>> Starting seed {seed} for {model_name} <<")
            
            # 1. Training (will save weights in runs/maskrcnn_seed_X/weights/best.pt)
            train_maskrcnn_model(dataset=dataset, data_yaml=data_yaml, seed=seed, epochs=20)
            
            # 2. Evaluation
            run_dir = os.path.abspath(f"runs{dataset}/maskrcnn_seed_{seed}")
            best_weights = os.path.join(run_dir, "weights", "best.pt")
            eval_metrics = evaluate_maskrcnn_model_unified(model_path=best_weights, data_yaml=data_yaml, seed=seed)
            
            metrics_row = {"seed": seed}
            metrics_row.update(eval_metrics)
            all_eval_metrics.append(metrics_row)
            
            # 3. Save CSV
            eval_csv_path = f"results{dataset}/{model_name}_evaluation_metrics_5_seeds.csv"
            pd.DataFrame(all_eval_metrics).to_csv(eval_csv_path, index=False)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        print(f"\n--- All seeds for {model_name} completed ---")

#'''
if __name__ == "__main__":
    main_maskrcnn(dataset='_a1')
    main_maskrcnn(dataset='_a2')
    main_maskrcnn(dataset='_a3')
    #'''
