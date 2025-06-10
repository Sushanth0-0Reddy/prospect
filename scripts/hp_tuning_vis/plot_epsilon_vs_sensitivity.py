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

from src.utils.io import var_to_str # Corrected import

FAIL_CODE = -1
L2_REG_DEFAULT = 1.0
SHIFT_COST_DEFAULT = 1.0

def parse_path_for_hyperparams_for_vis(path_str):
    """
    Parses hyperparameters from a directory path string for visualization purposes.
    Example DP: .../results_dp/eps_2.000000/acsincome/l2_reg_...obj_extremile.../bs_128_clip_1.0_lr_0.001_noise_multiplier_X.X.../seed_1.p
    """
    path_str = path_str.replace("\\", "/")
    hyperparams = {
        "lr": None, "batch_size": None, "clip_threshold": None, "noise_multiplier": None,
        "epsilon": None, "optimizer_type": None, "dataset": None, "objective": None,
        "full_config_dir_name": None
    }
    config_part = None

    if "results_dp/" in path_str:
        hyperparams["optimizer_type"] = "dp_sgd"
        match = re.search(r"results_dp/eps_([\d\.]+)/([^/]+)/([^/]+)/([^/]+)/seed_\d+\.p", path_str)
        if match:
            try: hyperparams["epsilon"] = float(match.group(1))
            except ValueError: pass
            hyperparams["dataset"] = match.group(2)
            hyperparams["full_config_dir_name"] = f"{match.group(3)}/{match.group(4)}"
            config_part = match.group(4)
            obj_match_outer = re.search(r"objective_(\w+)_shift_cost", match.group(3))
            if obj_match_outer: hyperparams["objective"] = obj_match_outer.group(1)
            
            ct_match = re.search(r"clip_threshold_([\d\.eE\+\-]+)", config_part)
            if ct_match: 
                try: hyperparams["clip_threshold"] = float(ct_match.group(1))
                except ValueError: pass
            nm_match = re.search(r"noise_multiplier_([\d\.eE\+\-]+)", config_part)
            if nm_match:
                try: hyperparams["noise_multiplier"] = float(nm_match.group(1))
                except ValueError: pass
    elif "results_sgd/" in path_str: # Included for completeness, though not used for this specific plot
        hyperparams["optimizer_type"] = "sgd"
        match = re.search(r"results_sgd/([^/]+)/([^/]+)/([^/]+)/seed_\d+\.p", path_str)
        if match:
            hyperparams["dataset"] = match.group(1)
            hyperparams["full_config_dir_name"] = f"{match.group(2)}/{match.group(3)}"
            config_part = match.group(3)
            obj_match_outer = re.search(r"objective_(\w+)_shift_cost", match.group(2))
            if obj_match_outer: hyperparams["objective"] = obj_match_outer.group(1)
    else:
        return hyperparams 

    if config_part: # Common for SGD and DP-SGD if config_part was set
        lr_match = re.search(r"lr_([\d\.eE\+\-]+)", config_part)
        if lr_match: 
            try: hyperparams["lr"] = float(lr_match.group(1))
            except ValueError: pass
        bs_match = re.search(r"batch_size_(\d+)", config_part)
        if bs_match: 
            try: hyperparams["batch_size"] = int(bs_match.group(1))
            except ValueError: pass

    if not hyperparams["objective"] and hyperparams["full_config_dir_name"]:
        obj_fallback = re.search(r"objective_(\w+)", hyperparams["full_config_dir_name"])
        if obj_fallback : hyperparams["objective"] = obj_fallback.group(1)
        
    return hyperparams

def get_model_cfg_for_lbfgs(dataset_name, objective_name, loss_type):
    n_class = None
    if dataset_name == "iwildcam": n_class = 60
    elif dataset_name == "amazon": n_class = 5
    
    cfg = {
        "objective": objective_name, "l2_reg": L2_REG_DEFAULT,
        "shift_cost": SHIFT_COST_DEFAULT, "loss": loss_type
    }
    if n_class is not None:
        cfg["n_class"] = n_class
    return cfg

def find_lbfgs_optimal_loss(base_results_dir, dataset_name, objective_name, loss_type):
    lbfgs_model_cfg = get_model_cfg_for_lbfgs(dataset_name, objective_name, loss_type)
    model_cfg_str = var_to_str(lbfgs_model_cfg)
    lbfgs_path = os.path.join(base_results_dir, "results_lbfgs", dataset_name, model_cfg_str, "lbfgs_min_loss.p")
    lbfgs_path = lbfgs_path.replace("\\", "/")
    print(f"Attempting L-BFGS optimal loss from: {lbfgs_path}")
    if not os.path.exists(lbfgs_path): return None
    try:
        with open(lbfgs_path, "rb") as f: min_loss = pickle.load(f)
        print(f"Loaded L-BFGS optimal loss: {min_loss}")
        return min_loss
    except Exception as e: return None

def find_best_hyperparams_for_epsilon(base_results_dir, dataset_name, objective_name, target_epsilon, l_opt):
    best_run_hyperparams = None
    min_final_suboptimality = float('inf')
    search_path = os.path.join(base_results_dir, f"results_dp/eps_{target_epsilon:.6f}", dataset_name)
    print(f"Searching for best DP-SGD run (eps={target_epsilon}) in: {search_path}")

    for root, dirs, files in os.walk(search_path):
        for file_name in files:
            if file_name.startswith("seed_") and file_name.endswith(".p"):
                file_path = os.path.join(root, file_name)
                parsed_h = parse_path_for_hyperparams_for_vis(file_path)
                if not (parsed_h["objective"] == objective_name and parsed_h["optimizer_type"] == "dp_sgd" and \
                        parsed_h["epsilon"] is not None and abs(parsed_h["epsilon"] - target_epsilon) < 1e-5):
                    continue
                try:
                    with open(file_path, "rb") as f: data = pickle.load(f)
                    if isinstance(data, int) and data == FAIL_CODE: continue
                    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], pd.DataFrame) and \
                       not data["metrics"].empty and "train_loss" in data["metrics"].columns:
                        metrics_df = data["metrics"]
                        initial_loss, final_loss = metrics_df["train_loss"].iloc[0], metrics_df["train_loss"].iloc[-1]
                        if initial_loss is None or final_loss is None or l_opt is None or (initial_loss - l_opt) == 0: continue
                        current_final_subopt = (final_loss - l_opt + 1e-9) / (initial_loss - l_opt + 1e-9)
                        if current_final_subopt < min_final_suboptimality:
                            min_final_suboptimality = current_final_subopt
                            best_run_hyperparams = parsed_h
                except Exception: continue
    
    if best_run_hyperparams:
         print(f"Best run for DP-SGD (eps={target_epsilon}) on {dataset_name}/{objective_name}: final_subopt={min_final_suboptimality:.3e}, C={best_run_hyperparams.get('clip_threshold')}, Sigma={best_run_hyperparams.get('noise_multiplier')}")
    return best_run_hyperparams

def load_best_config_from_file(base_results_dir, dataset_name, objective_name, loss_type, target_epsilon, l2_reg_val, shift_cost_val):
    n_c = None
    if dataset_name.lower() == "iwildcam": n_c = 60
    elif dataset_name.lower() == "amazon": n_c = 5
    # For 'acsincome' and others not specified, n_c remains None by default.

    model_cfg_dict = {
        "objective": objective_name,
        "l2_reg": l2_reg_val,
        "loss": loss_type,
        "n_class": n_c,
        "shift_cost": shift_cost_val
    }
    model_config_s = var_to_str(model_cfg_dict)
    optimizer_subdir = "dp_sgd" # Standard for DP results

    path_to_best_cfg = os.path.join(
        base_results_dir,
        f"results_dp/eps_{target_epsilon:.6f}",
        dataset_name,
        model_config_s,
        optimizer_subdir,
        "best_cfg.p"
    )
    path_to_best_cfg = path_to_best_cfg.replace("\\", "/") # Normalize

    print(f"Attempting to load best config from: {path_to_best_cfg}")
    if not os.path.exists(path_to_best_cfg):
        print(f"  best_cfg.p not found at: {path_to_best_cfg}")
        return None
    try:
        with open(path_to_best_cfg, "rb") as f:
            best_cfg_data = pickle.load(f)
        print(f"  Successfully loaded best_cfg.p for eps={target_epsilon:.6f} from {dataset_name}/{objective_name}.")
        return best_cfg_data  # This is the hyperparameter dictionary
    except Exception as e:
        print(f"  Error loading best_cfg.p from {path_to_best_cfg}: {e}")
        return None

def plot_epsilon_vs_sensitivity_metric(base_results_dir, dataset_name, objective_name, loss_type, epsilon_values_to_plot, output_dir_path, l2_reg_val, shift_cost_val):
    eps_plot_points = []
    cs_product_plot_points = []

    for eps_val in epsilon_values_to_plot:
        # MODIFIED: Load hyperparams from best_cfg.p instead of searching seeds
        best_hparams = load_best_config_from_file(
            base_results_dir, dataset_name, objective_name, loss_type, eps_val, l2_reg_val, shift_cost_val
        )
        
        if best_hparams and isinstance(best_hparams, dict):
            C = best_hparams.get("clip_threshold")
            sigma = best_hparams.get("noise_multiplier")

            if C is not None and sigma is not None:
                eps_plot_points.append(eps_val)
                cs_product_plot_points.append(C * sigma)
                print(f"Epsilon: {eps_val:.1f}, From best_cfg.p -> C: {C}, Sigma: {sigma:.2e}, C*Sigma: {C*sigma:.2e}")
            else:
                print(f"Warning: Missing C ({C}) or Sigma ({sigma}) in loaded best_cfg.p for epsilon {eps_val} for {dataset_name}/{objective_name}.")
        else:
            print(f"Warning: No best_cfg.p data or required params not found for epsilon {eps_val} for {dataset_name}/{objective_name}.")

    if not eps_plot_points:
        print(f"No data points to plot for {dataset_name}/{objective_name}.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(eps_plot_points, cs_product_plot_points, marker='o', linestyle='-')
    plt.xlabel(r"Target Epsilon ($\epsilon$)")
    plt.ylabel("Effective Sensitivity (Clip Norm $\times$ Noise Multiplier)")
    plt.title(f"Effective Sensitivity vs. Epsilon for {dataset_name.capitalize()} ({objective_name.capitalize()})")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(output_dir_path, exist_ok=True)
    plot_filename = os.path.join(output_dir_path, f"epsilon_vs_sensitivity_{dataset_name}_{objective_name}.png")
    
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot effective sensitivity (C*sigma) vs. epsilon based on best_cfg.p files.")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (default: acsincome)")
    parser.add_argument("--objective", type=str, default="extremile", help="Objective function (default: extremile)")
    parser.add_argument("--epsilons", type=str, default="2.0,4.0,10.0,100000.0", help="Comma-separated list of epsilon values to plot.")
    parser.add_argument("--base_results_dir", type=str, default="../../hp_tuning_experiments", help="Base directory for HP tuning results.")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR, help="Directory to save the plot (default: script\'s current directory).")
    # Add l2_reg and shift_cost to arguments as they are needed for model_config_str for path to best_cfg.p
    parser.add_argument("--l2_reg", type=float, default=L2_REG_DEFAULT, help=f"L2 regularization for model_cfg (default: {L2_REG_DEFAULT})")
    parser.add_argument("--shift_cost", type=float, default=SHIFT_COST_DEFAULT, help=f"Shift cost for model_cfg (default: {SHIFT_COST_DEFAULT})")

    args = parser.parse_args()

    resolved_base_results_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_results_dir))
    
    if os.path.isabs(args.output_dir):
        resolved_output_dir = args.output_dir
    else:
        resolved_output_dir = os.path.abspath(args.output_dir)

    if args.dataset.lower() in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
        loss_type_for_dataset = "squared_error"
    elif args.dataset.lower() == "diabetes": 
        loss_type_for_dataset = "binary_cross_entropy"
    else: 
        print(f"Warning: Unknown dataset \'{args.dataset}\' for loss type. Defaulting to \'squared_error\'.")
        loss_type_for_dataset = "squared_error"

    epsilons_list = [float(e.strip()) for e in args.epsilons.split(',') if e.strip()]

    print(f"Running plot for Dataset: {args.dataset}, Objective: {args.objective}")
    print(f"L2 Reg: {args.l2_reg}, Shift Cost: {args.shift_cost}") # ADDED print for new args
    print(f"Output will be saved in: {resolved_output_dir}")
    print(f"Reading base results from: {resolved_base_results_dir}")

    plot_epsilon_vs_sensitivity_metric(
        resolved_base_results_dir, 
        args.dataset, 
        args.objective, 
        loss_type_for_dataset, 
        epsilons_list,
        resolved_output_dir,
        args.l2_reg,      # Pass l2_reg
        args.shift_cost   # Pass shift_cost
    )

    print("\nScript finished.") 