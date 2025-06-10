import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Ensure src and scripts can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # For hp_tuning_vis imports

sys.path.append(SRC_DIR)
sys.path.append(SCRIPTS_DIR)

from utils.data import load_dataset
from utils.io import var_to_str # Assuming this is useful for parsing paths
# Need to import a function to parse hyperparams from path, similar to hp_tuning_vis
# Let's assume a similar helper can be created or imported if one from hp_tuning_vis is too specific

# Fairness metrics from Fairlearn
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

# Default objective for finding best runs (can be made a parameter later)
DEFAULT_OBJECTIVE = "extremile"
# Default L2 reg and shift cost, assuming they are consistent for model_cfg string generation
DEFAULT_L2_REG = 1.0
DEFAULT_SHIFT_COST = 1.0

FAIL_CODE = -1 # From training scripts

def parse_path_for_hyperparams_fairness(file_path, base_results_dir):
    """
    Parses a file_path from hp_tuning_experiments to extract hyperparameters.
    Adjusted to be more general if possible, or specific to fairness context.
    Example path structure:
    base_results_dir/results_{optimizer_type}/[eps_{epsilon}/]{dataset_name}/{model_config_str}/{optimizer_config_str}/seed_X.p
    """
    import re # Moved import re here as it's only used in this function
    parts = file_path.replace(base_results_dir, "").strip(os.sep).split(os.sep)
    hyperparams = {
        "lr": None, "batch_size": None, "clip_threshold": None, "epsilon": None,
        "optimizer_type": None, "dataset": None, "objective": DEFAULT_OBJECTIVE, # Default, might be in model_cfg_str
        "seed": None, "noise_multiplier": None
    }

    # Try to extract from path structure
    # results_dp/eps_10.000000/acsincome/l2_reg_1.00e+00_loss_squared_error_objective_extremile_shift_cost_1.00e+00/batch_size_64_clip_threshold_0.1_dataset_length_4000_lr_0.003_noise_multiplier_1.57_optimizer_dp_sgd_shift_cost_1.00e+00/seed_1.p
    # results_sgd/acsincome/l2_reg_1.00e+00_loss_squared_error_objective_extremile_shift_cost_1.00e+00/batch_size_32_dataset_length_4000_lr_0.03_optimizer_sgd_shift_cost_1.00e+00/seed_1.p
    
    if parts[0].startswith("results_"):
        hyperparams["optimizer_type"] = parts[0].split("_")[1]
    
    current_part_index = 1
    if hyperparams["optimizer_type"] == "dp_sgd" or hyperparams["optimizer_type"] == "dp": # accommodate results_dp
        if parts[current_part_index].startswith("eps_"):
            try:
                hyperparams["epsilon"] = float(parts[current_part_index].replace("eps_", ""))
            except ValueError:
                pass # Could not parse epsilon
            current_part_index += 1

    if len(parts) > current_part_index:
        hyperparams["dataset"] = parts[current_part_index]
        current_part_index += 1
    
    if len(parts) > current_part_index: # model_config_str
        model_cfg_str = parts[current_part_index]
        obj_match = re.search(r"objective_(\w+)", model_cfg_str)
        if obj_match:
            hyperparams["objective"] = obj_match.group(1)
        current_part_index += 1

    if len(parts) > current_part_index: # optimizer_config_str
        optim_cfg_str = parts[current_part_index]
        # Extract from optimizer_config_str
        lr_match = re.search(r"lr_([\d\.eE\+\-]+)", optim_cfg_str)
        if lr_match: hyperparams["lr"] = float(lr_match.group(1))
        
        bs_match = re.search(r"batch_size_(\d+)", optim_cfg_str)
        if bs_match: hyperparams["batch_size"] = int(bs_match.group(1))
        
        ct_match = re.search(r"clip_threshold_([\d\.eE\+\-]+)", optim_cfg_str)
        if ct_match: hyperparams["clip_threshold"] = float(ct_match.group(1))

        nm_match = re.search(r"noise_multiplier_([\d\.eE\+\-]+)", optim_cfg_str)
        if nm_match: hyperparams["noise_multiplier"] = float(nm_match.group(1))
        current_part_index += 1

    if len(parts) > current_part_index and parts[current_part_index].startswith("seed_"):
        try:
            hyperparams["seed"] = int(parts[current_part_index].replace("seed_", "").replace(".p", ""))
        except ValueError:
            pass
            
    return hyperparams

def find_best_run_for_config(base_results_dir, target_dataset, target_objective, optimizer_type, target_epsilon=None):
    """
    Finds the path to the seed file for the best run (lowest final training loss)
    for a given configuration.
    """
    best_loss = float('inf')
    best_run_file = None
    
    if optimizer_type == "dp_sgd":
        results_folder_name = "results_dp" # Corrected based on user's structure
    elif optimizer_type == "sgd":
        results_folder_name = "results_sgd"
    else:
        print(f"Warning: Unhandled optimizer_type '{optimizer_type}' for folder name. Using it directly.")
        results_folder_name = f"results_{optimizer_type}"

    search_path_base = os.path.join(base_results_dir, results_folder_name)

    if optimizer_type == "dp_sgd":
        if target_epsilon is None:
            print(f"ERROR: target_epsilon must be provided for optimizer_type dp_sgd")
            return None
        search_path = os.path.join(search_path_base, f"eps_{target_epsilon:.6f}", target_dataset)
    else: # sgd
        search_path = os.path.join(search_path_base, target_dataset)

    print(f"Searching for best {optimizer_type} (eps={target_epsilon}) run in: {search_path} for obj: {target_objective}")

    if not os.path.exists(search_path):
        print(f"  Search path does not exist: {search_path}")
        return None

    for root, dirs, files in os.walk(search_path):
        for file_name in files:
            if file_name.startswith("seed_") and file_name.endswith(".p"):
                file_path = os.path.join(root, file_name)
                # Basic parsing to check objective (can be refined with parse_path_for_hyperparams_fairness)
                if f"objective_{target_objective}" not in file_path: # Quick check
                    continue
                
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, int) and data == FAIL_CODE: continue # Skip diverged
                    if not isinstance(data, dict) or "metrics" not in data:
                        continue
                    
                    metrics_df = data["metrics"]
                    if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty or "train_loss" not in metrics_df.columns:
                        continue
                        
                    final_loss = metrics_df["train_loss"].iloc[-1]
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_run_file = file_path
                except Exception as e:
                    # print(f"Skipping file {file_path} due to error: {e}")
                    pass # Soft fail for individual file load errors
    
    if best_run_file:
        print(f"  Found best run: {best_run_file} with final loss: {best_loss:.4f}")
    else:
        print(f"  No suitable best run found for {optimizer_type} (eps={target_epsilon}) in {target_dataset}/{target_objective}")
        
    return best_run_file

def main():
    parser = argparse.ArgumentParser(description="Plot fairness metrics over epochs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., acsincome)")
    parser.add_argument("--objective", type=str, default=DEFAULT_OBJECTIVE, help=f"Objective function used for training (default: {DEFAULT_OBJECTIVE})")
    parser.add_argument("--epsilons", type=str, default="2.0,4.0,10.0,100000.0", help="Comma-separated list of epsilon values for DP-SGD to plot.")
    parser.add_argument("--sensitive_features", type=str, default="SEX,RAC1P", help="Comma-separated list of sensitive feature column names (case-insensitive match) (e.g., 'SEX,RAC1P' for acsincome)")
    parser.add_argument("--base_results_dir", type=str, default="../../hp_tuning_experiments", help="Base directory where hp_tuning_experiments results are stored.")

    args = parser.parse_args()

    # Prepare output directory
    output_dir = os.path.join(SCRIPT_DIR, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Resolve base results directory to absolute path
    base_results_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_results_dir))
    print(f"Reading results from: {base_results_dir}")

    # Load dataset (X, y, and Z for sensitive features)
    print(f"Loading dataset: {args.dataset}...")
    try:
        # load_dataset returns X_train, y_train, X_test, y_test
        # For fairness evaluation during training, we use the training portions.
        # data_path for load_dataset is relative to where load_dataset itself is, 
        # or if it handles paths intelligently, it might be workspace relative.
        # Given src/utils/data.py, and script in scripts/fairness_vis, ../../data/ seems correct.
        X_train, y_train, X_test, y_test = load_dataset(
            dataset=args.dataset, 
            data_path=os.path.join(SCRIPT_DIR, "../../data") # Path relative to this script
        )
        
        X_full = X_train 
        y_full = y_train

        # Load sensitive features (Z_train) separately
        # Expecting sensitive features for training data in 'metadata_tr.csv'
        z_train_path = os.path.join(SCRIPT_DIR, "../../data", args.dataset, "metadata_tr.csv")
        
        if not os.path.exists(z_train_path):
            raise FileNotFoundError(f"Sensitive features file metadata_tr.csv not found at {z_train_path}. Please ensure it exists.")
        else:
            print(f"Loading sensitive features from {z_train_path} as Z_full.")
            Z_full = pd.read_csv(z_train_path)

        if not isinstance(Z_full, pd.DataFrame):
             raise ValueError(f"Loaded sensitive features (Z_full) for dataset {args.dataset} is not a pandas DataFrame. Type: {type(Z_full)}. Please check data loading for Z.")
        
        print(f"Dataset loaded. X_full shape: {X_full.shape if hasattr(X_full, 'shape') else type(X_full)}, y_full shape: {y_full.shape if hasattr(y_full, 'shape') else type(y_full)}, Z_full columns: {Z_full.columns.tolist()}")

    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        sys.exit(1)

    requested_sensitive_features_arg = [s.strip() for s in args.sensitive_features.split(',') if s.strip()]
    z_full_cols_lower_map = {col.lower(): col for col in Z_full.columns}
    actual_column_names_for_analysis = []

    for req_sf_arg in requested_sensitive_features_arg:
        req_sf_lower = req_sf_arg.lower()
        if req_sf_lower in z_full_cols_lower_map:
            actual_column_names_for_analysis.append(z_full_cols_lower_map[req_sf_lower])
        else:
            print(f"Warning: Requested sensitive feature '{req_sf_arg}' not found in Z_full columns: {Z_full.columns.tolist()} (available map keys: {list(z_full_cols_lower_map.keys())}).")

    if not actual_column_names_for_analysis:
        print(f"No valid sensitive features to analyze from '{args.sensitive_features}' for dataset {args.dataset}. Available Z_full columns: {Z_full.columns.tolist()}")
        sys.exit(1)
    print(f"Will analyze fairness for sensitive features (actual column names): {actual_column_names_for_analysis}")

    # Configurations to plot
    plot_configs = []
    plot_configs.append({"label": "SGD", "optimizer_type": "sgd", "epsilon": None, "color": "black", "linestyle": "-"}) 
    
    epsilons = [float(e.strip()) for e in args.epsilons.split(',') if e.strip()]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(epsilons))) # Example color map
    linestyles = [":", "--", "-."]
    for i, eps in enumerate(epsilons):
        plot_configs.append({
            "label": rf"DP-SGD ($\epsilon={eps:.1f}$)", 
            "optimizer_type": "dp_sgd", 
            "epsilon": eps, 
            "color": colors[i],
            "linestyle": linestyles[i % len(linestyles)]
        })

    # Fairness metrics to compute and plot
    # Each item is a tuple: (metric_function, metric_name_for_plot, higher_is_worse_flag)
    fairness_metrics_to_plot = [
        (demographic_parity_difference, "Demographic Parity Difference", True),
        # (demographic_parity_ratio, "Demographic Parity Ratio", False), # Ratio near 1 is good
        (equalized_odds_difference, "Equalized Odds Difference", True),
        # (equalized_odds_ratio, "Equalized Odds Ratio", False), # Ratio near 1 is good
    ]

    # --- Main data collection and plotting loop ---
    for sf_name_in_df in actual_column_names_for_analysis: # Use the actual column name from Z_full
        print(f"\n--- Analyzing Sensitive Feature: {sf_name_in_df} ---")
        sensitive_feature_column = Z_full[sf_name_in_df]

        for metric_func, metric_plot_name, higher_is_worse in fairness_metrics_to_plot:
            plt.figure(figsize=(12, 7))
            
            print(f"  Metric: {metric_plot_name}")

            for config in plot_configs:
                print(f"    Processing: {config['label']}")
                best_run_path = find_best_run_for_config(
                    base_results_dir, args.dataset, args.objective,
                    config["optimizer_type"], config["epsilon"]
                )

                if not best_run_path:
                    print(f"      No best run found for {config['label']}. Skipping.")
                    continue
                
                try:
                    with open(best_run_path, "rb") as f:
                        run_data = pickle.load(f)
                    if not isinstance(run_data, dict) or "metrics" not in run_data:
                        print(f"      Invalid data in {best_run_path}. Skipping.")
                        continue
                    metrics_df = run_data["metrics"]
                    if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty or "iterations" not in metrics_df.columns:
                        print(f"      Metrics or iterations column missing in {best_run_path}. Skipping.")
                        continue

                    epoch_fairness_values = []
                    epochs = metrics_df["epoch"].values[1:] # Skip epoch -1 (initial state)
                    
                    # Ensure X_full and y_full are tensors for matmul
                    if not isinstance(X_full, torch.Tensor):
                        X_tensor = torch.tensor(X_full, dtype=torch.float64) 
                    else:
                        X_tensor = X_full.double() # Ensure double for matmul
                    if not isinstance(y_full, torch.Tensor):
                        y_tensor = torch.tensor(y_full, dtype=torch.int64) # Fairlearn expects int labels
                    else:
                        y_tensor = y_full.long()

                    # Ensure y_tensor for fairlearn is binary (0 or 1)
                    unique_labels = torch.unique(y_tensor)
                    if not ( (len(unique_labels) == 2 and ( (0. in unique_labels and 1. in unique_labels) or (-1. in unique_labels and 1. in unique_labels) )) or (len(unique_labels) == 1 and (0. in unique_labels or 1. in unique_labels or -1. in unique_labels)) ):
                        print(f"        Warning: y_tensor labels are {unique_labels}. Forcing to 0/1 for fairlearn based on min/max.")
                        min_val, max_val = torch.min(unique_labels), torch.max(unique_labels)
                        if len(unique_labels) > 2 or min_val == max_val: # more than 2 labels, or all same label.
                             print(f"        Cannot reliably binarize labels {unique_labels}. Skipping metric calculation for this run.")
                             # Fill with NaNs for this run and continue to next config
                             epoch_fairness_values = [np.nan] * len(epochs)
                             plt.plot(epochs, epoch_fairness_values, label=config["label"], color=config["color"], linestyle=config["linestyle"], marker='x', alpha=0.5) # Mark as problematic
                             continue # to next config in plot_configs
                        # Binarize: map min to 0, max to 1
                        y_tensor = (y_tensor == max_val).long()
                    else:
                        # If labels are -1, 1, map them to 0, 1 for consistency with (logits >= 0).long()
                        if -1. in unique_labels and 1. in unique_labels and 0. not in unique_labels:
                            y_tensor = ((y_tensor + 1) / 2).long()
                        else:
                             y_tensor = y_tensor # Already 0/1 or single value that's ok

                    for idx, epoch_num in enumerate(tqdm(epochs, desc=f"      Epochs for {config['label']}", leave=False)):
                        weights_at_epoch = metrics_df["iterations"].iloc[idx + 1] 
                        if not isinstance(weights_at_epoch, torch.Tensor):
                            print(f"        Weight for epoch {epoch_num} is not a tensor (type: {type(weights_at_epoch)}). Skipping epoch.")
                            epoch_fairness_values.append(np.nan) 
                            continue
                        
                        # Make predictions
                        # Assuming weights are for a linear model and X_tensor has features
                        logits = X_tensor @ weights_at_epoch.double() # Ensure dtype compatibility
                        y_pred = (logits >= 0).long() # Binary predictions (0 or 1)
                        
                        # Calculate fairness metric
                        # Ensure sensitive_feature_column is in a compatible format (e.g., list, numpy array)
                        sf_values_for_metric = sensitive_feature_column.values 
                        
                        try:
                            metric_val = metric_func(
                                y_tensor.numpy(), # fairlearn usually expects numpy arrays
                                y_pred.numpy(),
                                sensitive_features=sf_values_for_metric
                            )
                            epoch_fairness_values.append(metric_val)
                        except Exception as e_metric:
                            print(f"        Error calculating metric for epoch {epoch_num} ({config['label']}): {e_metric}")
                            epoch_fairness_values.append(np.nan)

                    plt.plot(epochs, epoch_fairness_values, label=config["label"], color=config["color"], linestyle=config["linestyle"], marker='.')
                
                except Exception as e_run:
                    print(f"      Error processing run {best_run_path}: {e_run}")
                    import traceback
                    traceback.print_exc()

            plt.xlabel("Epoch")
            plt.ylabel(metric_plot_name)
            title_str = f"{metric_plot_name} vs. Epoch\nDataset: {args.dataset.capitalize()}, Objective: {args.objective.capitalize()}, Sensitive Feature: {sf_name_in_df.capitalize()}"
            if higher_is_worse:
                title_str += " (Lower is Better)"
            else:
                title_str += " (Closer to 1 or 0 is Better for Ratio/Diff respectively)" # Adjust as needed for ratios
            plt.title(title_str)
            plt.legend()
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"fairness_{args.dataset}_{args.objective}_{sf_name_in_df.replace(' ', '_')}_{metric_plot_name.replace(' ', '_')}.png")
            plt.savefig(plot_filename)
            print(f"    Plot saved to {plot_filename}")
            plt.close()

    print("\nFairness plotting script finished.")

if __name__ == "__main__":
    main() 