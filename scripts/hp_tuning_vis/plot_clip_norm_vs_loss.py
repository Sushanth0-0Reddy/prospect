import os
import sys
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse # Added for command-line arguments

# Ensure src can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.append(SRC_DIR)

# Assuming plot_epsilon_vs_sensitivity.py is in the same directory and contains these helpers:
from .plot_epsilon_vs_sensitivity import parse_path_for_hyperparams_for_vis, find_lbfgs_optimal_loss, get_model_cfg_for_lbfgs

FAIL_CODE = -1

def find_min_loss_for_clip_norms(base_results_dir, dataset_name, objective_name, target_epsilon):
    """
    For a fixed epsilon, finds the minimum final loss achieved for each unique clip_threshold.
    """
    min_loss_per_clip_norm = defaultdict(lambda: float('inf'))
    runs_for_clip_norm = defaultdict(list)

    search_path = os.path.join(base_results_dir, f"results_dp/eps_{target_epsilon:.6f}", dataset_name)
    print(f"Searching for DP-SGD runs (eps={target_epsilon}) in: {search_path} to analyze loss vs clip_norm")

    for root, dirs, files in os.walk(search_path):
        for file_name in files:
            if file_name.startswith("seed_") and file_name.endswith(".p"):
                file_path = os.path.join(root, file_name)
                parsed_h = parse_path_for_hyperparams_for_vis(file_path)

                # Filter for the correct objective, optimizer type, and epsilon
                if not (parsed_h["objective"] == objective_name and \
                        parsed_h["optimizer_type"] == "dp_sgd" and \
                        parsed_h["epsilon"] is not None and abs(parsed_h["epsilon"] - target_epsilon) < 1e-5 and \
                        parsed_h["clip_threshold"] is not None):
                    continue
                
                try:
                    with open(file_path, "rb") as f: data = pickle.load(f)
                    
                    if isinstance(data, int) and data == FAIL_CODE: continue # Skip diverged runs
                    
                    if isinstance(data, dict) and "metrics" in data and \
                       isinstance(data["metrics"], pd.DataFrame) and not data["metrics"].empty and \
                       "train_loss" in data["metrics"].columns:
                        metrics_df = data["metrics"]
                        final_loss = metrics_df["train_loss"].iloc[-1]
                        
                        current_clip_norm = parsed_h["clip_threshold"]
                        runs_for_clip_norm[current_clip_norm].append(final_loss)
                        
                        if final_loss < min_loss_per_clip_norm[current_clip_norm]:
                            min_loss_per_clip_norm[current_clip_norm] = final_loss
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")
                    continue
    
    if not min_loss_per_clip_norm:
        print(f"No successful runs found for epsilon {target_epsilon} to analyze clip_norm vs loss.")
        return None
    
    # Sort by clip_norm for plotting
    sorted_clip_norms = sorted(min_loss_per_clip_norm.keys())
    sorted_min_losses = [min_loss_per_clip_norm[cn] for cn in sorted_clip_norms]
    
    print("Min loss per clip norm:")
    for cn, loss_val in zip(sorted_clip_norms, sorted_min_losses):
        print(f"  Clip Norm: {cn:.2f}, Min Final Loss: {loss_val:.4f}, (based on {len(runs_for_clip_norm[cn])} runs)")
        
    return sorted_clip_norms, sorted_min_losses

def plot_clip_norm_vs_min_loss(base_results_dir, dataset_name, objective_name, target_epsilon, loss_type, output_dir_path):
    result = find_min_loss_for_clip_norms(base_results_dir, dataset_name, objective_name, target_epsilon)
    
    if result is None:
        print(f"No data to plot for clip_norm vs loss for epsilon {target_epsilon} on {dataset_name}/{objective_name}.")
        return
    
    clip_norms, min_losses = result
    
    if not clip_norms or not min_losses: # Double check, though find_min_loss_for_clip_norms should handle this
        print(f"No data points returned from find_min_loss_for_clip_norms for epsilon {target_epsilon} on {dataset_name}/{objective_name}.")
        return

    l_opt = find_lbfgs_optimal_loss(base_results_dir, dataset_name, objective_name, loss_type)

    plt.figure(figsize=(10, 6))
    plt.plot(clip_norms, min_losses, marker='o', linestyle='-')
    
    if l_opt is not None:
        plt.axhline(y=l_opt, color='r', linestyle='--', label=f'L-BFGS $L_{{opt}}$ ({l_opt:.4f})')
        plt.legend()

    plt.xlabel("Clip Norm Threshold (C)")
    plt.ylabel("Minimum Final Training Loss Achieved")
    plt.title(f"Min Loss vs. Clip Norm for DP-SGD ($\epsilon={target_epsilon}$)\nDataset: {dataset_name.capitalize()}, Objective: {objective_name.capitalize()}")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    
    # Use the provided output_dir_path for saving
    os.makedirs(output_dir_path, exist_ok=True) # Ensure the directory exists
    plot_filename = os.path.join(output_dir_path, f"clip_norm_vs_loss_eps{target_epsilon:.0f}_{dataset_name}_{objective_name}.png")
    
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Removed or commented out
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot minimum final loss vs. clip norm for DP-SGD at specified epsilons.")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (default: acsincome)")
    parser.add_argument("--objective", type=str, default="extremile", help="Objective function (default: extremile)")
    parser.add_argument("--epsilons", type=str, default="2.0,4.0,10.0,100000.0", help="Comma-separated list of epsilon values to generate plots for.")
    parser.add_argument("--base_results_dir", type=str, default="../../hp_tuning_experiments", help="Base directory for HP tuning results.")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR, help="Directory to save the plots (default: script's current directory).")

    args = parser.parse_args()

    resolved_base_results_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_results_dir))
    
    if not os.path.isabs(args.output_dir):
        resolved_output_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.output_dir))
    else:
        resolved_output_dir = args.output_dir

    if args.dataset.lower() in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
        loss_type_for_dataset = "squared_error"
    elif args.dataset.lower() == "diabetes": 
        loss_type_for_dataset = "binary_cross_entropy"
    else: 
        print(f"Warning: Unknown dataset '{args.dataset}' for loss type. Defaulting to 'squared_error'.")
        loss_type_for_dataset = "squared_error" 

    epsilons_list_to_analyze = [float(e.strip()) for e in args.epsilons.split(',') if e.strip()]

    print(f"Generating Clip Norm vs Loss plots for Dataset: {args.dataset}, Objective: {args.objective}")
    print(f"Output will be saved in: {resolved_output_dir}")
    print(f"Reading base results from: {resolved_base_results_dir}")

    for eps_val in epsilons_list_to_analyze:
        print(f"\n--- Generating plot for Epsilon = {eps_val} ---")
        plot_clip_norm_vs_min_loss(
            resolved_base_results_dir, 
            args.dataset, 
            args.objective, 
            eps_val, 
            loss_type_for_dataset,
            resolved_output_dir
        )
    print("\nScript finished.") 