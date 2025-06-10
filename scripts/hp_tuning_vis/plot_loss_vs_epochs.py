import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from collections import defaultdict # Not strictly needed anymore
# import re # Not strictly needed anymore

# Ensure src and scripts can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.append(SRC_DIR)
sys.path.append(SCRIPTS_DIR)

from src.utils.io import var_to_str # For creating model_config_str

# Default objective for argument parsing
DEFAULT_OBJECTIVE = "extremile"
# Constants for model_cfg, assuming these are fixed for the "best" runs you want to compare
DEFAULT_L2_REG = 1.0
DEFAULT_SHIFT_COST = 1.0

def get_loss_type_for_dataset(dataset_name):
    """ Returns the default loss type string based on dataset name. """
    if dataset_name.lower() == "acsincome":
        return "squared_error"
    # Add other datasets and their typical loss functions here if needed
    # For example, diabetes often uses binary_cross_entropy
    elif dataset_name.lower() == "diabetes":
        return "binary_cross_entropy"
    else:
        # Fallback, or raise an error if the dataset isn't recognized
        print(f"Warning: Unknown dataset '{dataset_name}' for loss type. Defaulting to 'binary_cross_entropy'.")
        return "binary_cross_entropy"

def main():
    parser = argparse.ArgumentParser(description="Plot training loss vs. epochs for SGD and DP-SGD from best_traj.p files.")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (default: acsincome)")
    parser.add_argument("--objective", type=str, default=DEFAULT_OBJECTIVE, help=f"Objective function (default: {DEFAULT_OBJECTIVE})")
    parser.add_argument("--epsilons", type=str, default="2.0,4.0,10.0,100000.0", help="Comma-separated list of epsilon values for DP-SGD.")
    parser.add_argument("--base_results_dir", type=str, default="../../hp_tuning_experiments", help="Base directory for HP tuning results.")
    parser.add_argument("--output_subdir", type=str, default="acsincome", help="Subdirectory within scripts/hp_tuning_vis to save plots.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs to plot.")
    parser.add_argument("--l2_reg", type=float, default=DEFAULT_L2_REG, help=f"L2 regularization value for model_cfg (default: {DEFAULT_L2_REG})")
    parser.add_argument("--shift_cost", type=float, default=DEFAULT_SHIFT_COST, help=f"Shift cost value for model_cfg (default: {DEFAULT_SHIFT_COST})")


    args = parser.parse_args()

    # Prepare output directory
    output_base = os.path.join(SCRIPT_DIR, args.output_subdir)
    os.makedirs(output_base, exist_ok=True)
    print(f"Saving plots to: {output_base}")

    # Resolve base results directory to absolute path
    base_results_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_results_dir))
    print(f"Reading results from: {base_results_dir}")

    # Determine loss type based on dataset
    loss_type = get_loss_type_for_dataset(args.dataset)

    # Construct the common model_cfg part
    # Note: n_class is usually None for binary classification or regression objectives like extremile
    model_cfg = {
        "objective": args.objective,
        "l2_reg": args.l2_reg,
        "loss": loss_type,
        "n_class": None, 
        "shift_cost": args.shift_cost
    }
    model_config_str = var_to_str(model_cfg)
    print(f"Using model_config_str: {model_config_str}")

    plot_configs = []
    # SGD config
    plot_configs.append({
        "label": "SGD", "optimizer_type": "sgd", "epsilon": None, 
        "color": "black", "linestyle": "-", "marker": "."
    }) 
    
    # DP-SGD configs
    epsilons_list = [float(e.strip()) for e in args.epsilons.split(',') if e.strip()]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(epsilons_list))) 
    linestyles = [":", "--", "-.", (0, (3, 1, 1, 1))] 
    markers = ['o', 's', '^', 'D', 'v']

    for i, eps in enumerate(epsilons_list):
        plot_configs.append({
            "label": f"DP-SGD ($\epsilon={eps:.1f}$)", 
            "optimizer_type": "dp_sgd",
            "epsilon": eps, 
            "color": colors[i],
            "linestyle": linestyles[i % len(linestyles)],
            "marker": markers[i % len(markers)]
        })

    plt.figure(figsize=(12, 8))

    for config in plot_configs:
        print(f"Processing: {config['label']}")
        
        path_to_best_traj = None
        optimizer_subdir_name = None # To store 'sgd' or 'dp_sgd'

        if config["optimizer_type"] == "sgd":
            results_folder_name = "results_sgd"
            optimizer_subdir_name = "sgd" # Standard name for SGD optimizer subdir
            path_to_best_traj = os.path.join(base_results_dir, results_folder_name, args.dataset, model_config_str, optimizer_subdir_name, "best_traj.p")
        elif config["optimizer_type"] == "dp_sgd":
            results_folder_name = "results_dp" 
            optimizer_subdir_name = "dp_sgd" # Standard name for DP-SGD optimizer subdir
            epsilon_str = f"eps_{config['epsilon']:.6f}"
            path_to_best_traj = os.path.join(base_results_dir, results_folder_name, epsilon_str, args.dataset, model_config_str, optimizer_subdir_name, "best_traj.p")

        if not path_to_best_traj or not os.path.exists(path_to_best_traj):
            print(f"  best_traj.p not found at: {path_to_best_traj}. Skipping {config['label']}.")
            continue
        
        print(f"  Loading best_traj.p from: {path_to_best_traj}")
        try:
            with open(path_to_best_traj, "rb") as f:
                traj_df = pickle.load(f)
            
            if not isinstance(traj_df, pd.DataFrame) or traj_df.empty:
                print(f"  best_traj.p did not contain a valid DataFrame or is empty. Skipping {config['label']}.")
                continue
            
            # Expected columns in best_traj.p (based on plot_suboptimality_acsincome.py and typical analysis scripts)
            # 'average_train_loss' or 'train_loss'. 'epoch' might or might not be present.
            loss_col_name = None
            if "average_train_loss" in traj_df.columns:
                loss_col_name = "average_train_loss"
            elif "train_loss" in traj_df.columns: # Fallback if 'average_train_loss' is not present
                loss_col_name = "train_loss"
            else:
                print(f"  Could not find 'average_train_loss' or 'train_loss' column in {path_to_best_traj}. Skipping {config['label']}.")
                continue
            
            loss_to_plot = traj_df[loss_col_name].values
            
            # Epochs: If 'epoch' column exists, use it. Otherwise, generate from length.
            # The 'epoch' in 'best_traj.p' (if generated from analyze_hp_results.py style)
            # often corresponds to the actual epoch number from the training loop (e.g., 0 to max_epochs-1).
            # The seed_X.p files might have an initial -1 epoch, but best_traj.p usually has processed epochs.
            if 'epoch' in traj_df.columns:
                epochs_to_plot = traj_df["epoch"].values
            else:
                # Assuming loss values correspond to epochs 0, 1, 2, ...
                # This matches how plot_suboptimality_acsincome.py generates epochs if not directly available
                epochs_to_plot = np.arange(len(loss_to_plot)) 

            if args.max_epochs is not None:
                # Ensure epochs_to_plot is a numpy array for boolean indexing
                epochs_to_plot_np = np.array(epochs_to_plot)
                valid_indices = epochs_to_plot_np <= args.max_epochs
                epochs_to_plot = epochs_to_plot_np[valid_indices]
                loss_to_plot = loss_to_plot[valid_indices]

            if len(epochs_to_plot) > 0:
                plt.plot(epochs_to_plot, loss_to_plot, label=config["label"], 
                         color=config["color"], linestyle=config["linestyle"], 
                         marker=config["marker"], markersize=5, alpha=0.8)
            else:
                print(f"  No data points to plot for {config['label']} after epoch filtering (max_epochs={args.max_epochs}).")

        except Exception as e_run:
            print(f"  Error processing run {path_to_best_traj}: {e_run}")
            import traceback
            traceback.print_exc()

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (from best_traj.p)")
    plt.title(f"Training Loss vs. Epochs\nDataset: {args.dataset.capitalize()}, Objective: {args.objective.capitalize()}\nModel Config: {model_config_str}")
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_base, f"loss_vs_epochs_{args.dataset}_{args.objective}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

    print("\nScript finished.")

if __name__ == "__main__":
    main() 