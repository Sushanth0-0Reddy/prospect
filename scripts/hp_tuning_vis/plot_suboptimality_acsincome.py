import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Ensure src can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.append(SRC_DIR)

from utils.io import var_to_str

FAIL_CODE = -1
L2_REG_DEFAULT = 1.0
SHIFT_COST_DEFAULT = 1.0

def get_n_class(dataset_name):
    """ Helper to get n_class based on dataset name. """
    if dataset_name.lower() == "iwildcam": return 60
    elif dataset_name.lower() == "amazon": return 5
    return None

def get_model_cfg_for_lbfgs(dataset_name, objective_name, loss_type):
    """ Creates a model_cfg dict consistent with how lbfgs.py would form it. """
    return {
        "objective": objective_name,
        "l2_reg": L2_REG_DEFAULT,
        "shift_cost": SHIFT_COST_DEFAULT,
        "loss": loss_type,
        "n_class": get_n_class(dataset_name)
    }

def find_lbfgs_optimal_loss(base_results_dir, dataset_name, objective_name, loss_type):
    lbfgs_model_cfg = get_model_cfg_for_lbfgs(dataset_name, objective_name, loss_type)
    model_cfg_str = var_to_str(lbfgs_model_cfg)
    
    lbfgs_path = os.path.join(base_results_dir, "results_lbfgs", dataset_name, model_cfg_str, "lbfgs_min_loss.p")
    lbfgs_path = lbfgs_path.replace("\\", "/")
    print(f"Attempting to load L-BFGS optimal loss from: {lbfgs_path}")

    if not os.path.exists(lbfgs_path):
        print(f"Error: L-BFGS results file not found at {lbfgs_path}")
        return None
    try:
        with open(lbfgs_path, "rb") as f:
            min_loss = pickle.load(f)
        print(f"Loaded L-BFGS optimal loss: {min_loss}")
        return min_loss
    except Exception as e:
        print(f"Error loading L-BFGS results from {lbfgs_path}: {e}")
        return None

def load_best_trajectory_from_simple_path(base_results_dir, dataset_name, objective_name, loss_type, 
                                          optimizer_type, target_epsilon):
    """
    Loads the best_traj.p DataFrame from the simplified path structure
    created by analyze_hp_results.py.
    """
    model_cfg_dict = {
        "objective": objective_name,
        "l2_reg": L2_REG_DEFAULT,
        "shift_cost": SHIFT_COST_DEFAULT,
        "loss": loss_type,
        "n_class": get_n_class(dataset_name)
    }
    model_config_s = var_to_str(model_cfg_dict)

    results_folder_name = "results_sgd" if optimizer_type == "sgd" else "results_dp"
    simple_optim_subdir = "sgd" if optimizer_type == "sgd" else "dp_sgd"
    
    path_parts = [base_results_dir, results_folder_name]
    if optimizer_type == "dp_sgd":
        if target_epsilon is None:
            print(f"Warning: target_epsilon is None for dp_sgd. Cannot construct path.")
            return None
        path_parts.append(f"eps_{target_epsilon:.6f}")
    
    path_parts.extend([dataset_name, model_config_s, simple_optim_subdir, "best_traj.p"])
    path_to_traj = os.path.join(*path_parts)
    path_to_traj = path_to_traj.replace("\\", "/")

    print(f"Attempting to load best trajectory for {optimizer_type} (eps={target_epsilon}) from: {path_to_traj}")
    if not os.path.exists(path_to_traj):
        print(f"  best_traj.p not found at: {path_to_traj}")
        return None
    try:
        with open(path_to_traj, "rb") as f:
            traj_df = pickle.load(f)
        # Check for 'average_train_loss' first, then fall back to 'train_loss' for broader compatibility
        if isinstance(traj_df, pd.DataFrame) and "average_train_loss" in traj_df.columns and not traj_df.empty:
            print(f"  Successfully loaded best_traj.p (using 'average_train_loss') for {optimizer_type} (eps={target_epsilon}).")
            return traj_df["average_train_loss"].copy()
        elif isinstance(traj_df, pd.DataFrame) and "train_loss" in traj_df.columns and not traj_df.empty:
            print(f"  Successfully loaded best_traj.p (using 'train_loss') for {optimizer_type} (eps={target_epsilon}).")
            return traj_df["train_loss"].copy()
        else:
            print(f"  best_traj.p at {path_to_traj} is not a valid DataFrame or lacks 'average_train_loss' or 'train_loss' column.")
            if isinstance(traj_df, pd.DataFrame):
                print(f"    Available columns: {traj_df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"  Error loading or processing best_traj.p from {path_to_traj}: {e}")
        return None

def plot_suboptimality_curves(base_results_dir, dataset_name, objective_name, loss_type, output_dir_path):
    l_opt = find_lbfgs_optimal_loss(base_results_dir, dataset_name, objective_name, loss_type)
    if l_opt is None:
        print(f"Cannot proceed with plotting for {dataset_name}/{objective_name} as L_opt could not be determined.")
        return

    optimizer_configs = [
        {"type": "sgd", "epsilon": None, "label": r"SGD", "color": "black", "linestyle": "-"},
        {"type": "dp_sgd", "epsilon": 2.0, "label": r"DP-SGD ($\epsilon=2$)", "color": "blue", "linestyle": "--"},
        {"type": "dp_sgd", "epsilon": 4.0, "label": r"DP-SGD ($\epsilon=4$)", "color": "green", "linestyle": ":"},
        {"type": "dp_sgd", "epsilon": 10.0, "label": r"DP-SGD ($\epsilon=10$)", "color": "red", "linestyle": "-."},
        {"type": "dp_sgd", "epsilon": 100000.0, "label": r"DP-SGD ($\epsilon=10^5$)", "color": "purple", "linestyle": "--"},
    ]

    plt.figure(figsize=(12, 8))

    for config in optimizer_configs:
        print(f"Processing config: {config['label']}")
        train_loss_curve = load_best_trajectory_from_simple_path(
            base_results_dir, dataset_name, objective_name, loss_type,
            config["type"], config["epsilon"]
        )

        if train_loss_curve is not None and not train_loss_curve.empty:
            initial_loss = train_loss_curve.iloc[0]
            if l_opt is None:
                print(f"Skipping {config['label']} as L_opt is None.")
                continue
            if (initial_loss - l_opt) == 0:
                print(f"Warning: Initial loss equals L_opt for {config['label']}. Plotting suboptimality as 0.")
                suboptimality_curve = np.zeros_like(train_loss_curve.to_numpy(dtype=float))
            else:
                suboptimality_curve = (train_loss_curve.to_numpy(dtype=float) - l_opt + 1e-9) / (initial_loss - l_opt + 1e-9)
            
            epochs = np.arange(len(suboptimality_curve))
            plt.plot(epochs, suboptimality_curve, label=config["label"], color=config["color"], linestyle=config["linestyle"], linewidth=2.5)
        else:
            print(f"No trajectory data found for {config['label']}. Skipping.")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(r"Suboptimality ($(L_t - L_{opt}) / (L_0 - L_{opt})$)")
    plt.title(f"Suboptimality vs. Epochs for {dataset_name.capitalize()} ({objective_name.capitalize()})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(output_dir_path, exist_ok=True)
    plot_filename = os.path.join(output_dir_path, f"suboptimality_{dataset_name}_{objective_name}.png")
    
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot suboptimality curves for SGD and DP-SGD from pre-calculated best_traj.p files.")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (default: acsincome)")
    parser.add_argument("--objective", type=str, default="extremile", help="Objective function (default: extremile)")
    parser.add_argument("--base_results_dir", type=str, default="../../hp_tuning_experiments", help="Base directory for HP tuning results relative to script location.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the plot.")

    args = parser.parse_args()

    resolved_base_results_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_results_dir))
    
    if os.path.isabs(args.output_dir):
        resolved_output_dir = args.output_dir
    else:
        resolved_output_dir = os.path.abspath(args.output_dir)


    if args.dataset.lower() in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
        loss_for_dataset = "squared_error"
    elif args.dataset.lower() == "diabetes":
        loss_for_dataset = "binary_cross_entropy"
    else:
        print(f"Warning: Loss type for dataset '{args.dataset}' not explicitly defined, defaulting to 'squared_error'. Review if this is correct.")
        loss_for_dataset = "squared_error"

    print(f"Running suboptimality plot for Dataset: {args.dataset}, Objective: {args.objective}")
    print(f"Reading base results from: {resolved_base_results_dir}")
    print(f"Saving plot to: {resolved_output_dir}")

    plot_suboptimality_curves(resolved_base_results_dir, args.dataset, args.objective, loss_for_dataset, resolved_output_dir) 