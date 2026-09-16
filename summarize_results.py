#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np

def summarize_results(dataset, path):
    # Search deeply through all subfolders of "results" for the final evaluation CSVs
    csv_files = glob.glob(f"{path}/results{dataset}/**/*_evaluation_metrics_5_seeds.csv", recursive=True)

    # If the folder structure is flatter or files have different names, be safe and collect anything that matches
    if not csv_files:
        csv_files = glob.glob(f"{path}/results{dataset}/**/*.csv", recursive=True)
        # Keep only the aggregated evaluation CSVs
        csv_files = [f for f in csv_files if "evaluation_metrics_5_seeds.csv" in f]

    if not csv_files:
        print("No CSV files with evaluation metrics were found in the 'results' folder!")
        return

    all_data = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            # Extract the model name from the parent folder name
            # (since each model is stored in its own folder)
            model_name = os.path.basename(os.path.dirname(file_path))

            # If the path is something like "results/yolo11n-seg.yaml_evaluation...", use the filename instead
            if model_name.lower() in [f"results{dataset}", ".", ".."]:
                model_name = os.path.basename(file_path).split('_evaluation_metrics')[0]

            df['Model'] = model_name
            all_data.append(df)

        except Exception as e:
            print(f"Could not read file {file_path}. Error: {e}")

    if not all_data:
        return

    # Combine all loaded DataFrames into one large table
    combined_df = pd.concat(all_data, ignore_index=True)

    # Identify all numeric columns (which we can round and average)
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()

    if 'seed' in num_cols:
        num_cols.remove('seed')  # Seed should not be averaged

    # Remove YOLO-specific metrics that Mask R-CNN does not compute (without PR curve),
    # so the table matches exactly for both models:
    cols_to_drop = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'fitness']
    num_cols = [c for c in num_cols if c not in cols_to_drop]

    # Group by 'Model' and compute the mean and standard deviation
    agg_df = combined_df.groupby('Model')[num_cols].agg(['mean', 'std'])

    # Format the table so it is neat and readable.
    # We combine mean and standard deviation into one field per metric: "0.95 ± 0.02" for readability.

    # Flat header for simple data export (e.g. "metrics/mAP50(B)_mean")
    export_df = agg_df.copy()
    export_df.columns = [f"{col[0]}_{col[1]}" for col in export_df.columns]
    export_df = export_df.reset_index()

    # Readable header for display with +/- notation
    pretty_df = pd.DataFrame()
    pretty_df['Model'] = agg_df.index
    for metric in num_cols:
        # Round to 4 decimal places
        mean = agg_df[(metric, 'mean')].values
        std = agg_df[(metric, 'std')].values

        # Create the string "Mean ± Std" and handle NaNs and -1.0 (TorchMetrics error code)
        pretty_strs = []
        for m, s in zip(mean, std):
            if np.isnan(m) or m < -0.5:
                pretty_strs.append("N/A")
            else:
                s_val = 0.0 if np.isnan(s) else s
                pretty_strs.append(f"{m:.4f} ± {s_val:.4f}")

        pretty_df[metric] = pretty_strs

    # Save the raw data table (useful for other scripts)
    raw_output = f"{path}/results{dataset}/All_Models_Aggregated_Raw.csv"
    export_df.to_csv(raw_output, index=False)

    # Save the formatted table (ideal for viewing, papers, or Excel)
    pretty_output = f"{path}/results{dataset}/All_Models_Averaged_Pretty.csv"
    pretty_df.to_csv(pretty_output, index=False)

    print(f"\n--- Summary completed successfully! ---")
    print(f"Raw data saved in: {raw_output}")
    print(f"Formatted overview saved in: {pretty_output}\n")

    print("Preview of the results:")
    print(pretty_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=".", help="folder containing the results")
    args = parser.parse_args()

    summarize_results(dataset='_a1', path=args.path)
    summarize_results(dataset='_a2', path=args.path)
    summarize_results(dataset='_a3', path=args.path)
