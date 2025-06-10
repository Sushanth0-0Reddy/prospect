import argparse
import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fairlearn.metrics import demographic_parity_difference

# --- Add SRC_DIR to sys.path ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from utils.io import var_to_str, optim_str_to_dict_for_vis
    from datasets.acs_pytorch import load_acsincome_dataset_for_fairlearn
    from utils.exp_mgmt import find_best_optimizer_run_from_folder_list # May need adaptation
except ImportError as e:
    print(f"Error importing from src: {e}")
    sys.exit(1)

# --- Global Configs (adapt from plot_fairness_from_notebook.py) ---
HP_TUNING_DIR_PROSPECT = Path(os.getenv("HP_TUNING_DIR_PROSPECT", Path(__file__).resolve().parent.parent.parent / "hp_tuning_experiments"))
DEFAULT_SEED = 1
MODEL_CFG_SHARED = {'l2_reg': 1.0, 'shift_cost': 1.0, 'loss': 'squared_error', 'n_class': None} # n_class is for Fairlearn
OPTIM_CFG_SGD = { # Based on common SGD settings seen
    'optimizer': 'sgd', 'lr': 0.003, 'dataset_length': 4000,
    'batch_size': 128, 'shift_cost': 1.0, 'noise_multiplier': 0.0
}
EPSILON_DP_SGD_TO_PLOT = 2.0
OBJECTIVES_TO_PLOT = ["erm", "extremile", "esrm", "superquantile"]
MOVING_AVG_WINDOW = 5 # Optional: for smoothing, can be set to 1 for no smoothing

# --- Helper function to calculate moving average (from plot_fairness_from_notebook.py) ---
def moving_average(data, window_size):
    if not isinstance(data, (list, np.ndarray)) or not data:
        return np.array([])
    if window_size <= 0:
        return np.array([])
    np_data = np.array(data)
    if np_data.size == 0:
        return np.array([])
    if len(np_data) < window_size or window_size == 1:
        return np_data # Return original data if too short for convolution or window is 1
    
    # Calculate moving average using convolution; 'valid' mode ensures no padding
    smoothed = np.convolve(np_data, np.ones(window_size)/window_size, mode='valid')
    
    # To ensure the plotted line has the same length as original epochs for alignment,
    # we can pad the start of the smoothed data with NaNs or repeat the first valid value.
    # For simplicity here, we'll pad with the first valid smoothed value.
    # Or, more simply, ensure enough data points exist. If not, original data is returned.
    if smoothed.size == 0: # Should not happen if len(np_data) >= window_size > 0
        return np_data # Fallback
    return smoothed

# --- Helper function to get iterates (adapted from plot_fairness_from_notebook.py) ---
def get_iterates_data(
    optimizer_type, # "sgd" or "dp_sgd"
    base_optim_cfg_dict, # For SGD, this is OPTIM_CFG_SGD. For DP-SGD, this is the selected one.
    model_cfg_dict,
    dataset_name_arg,
    hp_tuning_dir,
    seed,
    epsilon_val=None # Only for DP-SGD
):
    print(f"Attempting to load iterates for: optimizer='{optimizer_type}', objective='{model_cfg_dict.get('objective')}', seed={seed}, epsilon={epsilon_val}")

    model_cfg_str = var_to_str(model_cfg_dict)
    optim_cfg_str = var_to_str(base_optim_cfg_dict)

    print(f"  Generated model_cfg_str for path: {model_cfg_str}")
    print(f"  Generated optim_cfg_str for path: {optim_cfg_str}")

    iterates_file_path = None
    if optimizer_type.lower() == "sgd":
        results_subfolder = "results_sgd"
        iterates_file_path = Path(hp_tuning_dir) / results_subfolder / dataset_name_arg / model_cfg_str / optim_cfg_str / f"seed_{seed}.p"
    elif optimizer_type.lower() == "dp_sgd" and epsilon_val is not None:
        results_subfolder = "results_dp"
        epsilon_folder = f"eps_{epsilon_val:.6f}"
        iterates_file_path = Path(hp_tuning_dir) / results_subfolder / epsilon_folder / dataset_name_arg / model_cfg_str / optim_cfg_str / f"seed_{seed}.p"
    else:
        print(f"ERROR: Invalid optimizer_type '{optimizer_type}' or missing epsilon for DP-SGD.")
        return None

    print(f"  Constructed iterates_file_path: {iterates_file_path}")

    if iterates_file_path and iterates_file_path.exists():
        try:
            with open(iterates_file_path, "rb") as f:
                loaded_data = pickle.load(f)
            print(f"  Successfully loaded iterates from: {iterates_file_path}")
            
            if isinstance(loaded_data, dict) and "metrics" in loaded_data and isinstance(loaded_data["metrics"], pd.DataFrame):
                if "iterations" in loaded_data["metrics"].columns:
                    return loaded_data["metrics"] # Return the DataFrame
                else:
                    print(f"  WARNING: 'iterations' column not found in metrics DataFrame for {iterates_file_path}")
            else:
                 print(f"  WARNING: Loaded data from {iterates_file_path} is not a dict with a 'metrics' DataFrame. Type: {type(loaded_data)}")
        except Exception as e:
            print(f"  ERROR: Failed to load or process iterates from {iterates_file_path}: {e}")
    else:
        print(f"  ERROR: Iterates file not found: {iterates_file_path}")
    return None

# --- Helper to select DP-SGD optimizer config (adapted from plot_fairness_from_notebook.py) ---
def select_dpsgd_optimizer_config_for_epsilon(
    hp_tuning_dir, dataset_name, model_cfg_dict, epsilon_val, target_objective_for_selection
):
    model_cfg_str = var_to_str(model_cfg_dict)
    epsilon_folder_name = f"eps_{epsilon_val:.6f}"
    base_path_for_eps_configs = Path(hp_tuning_dir) / "results_dp" / epsilon_folder_name / dataset_name / model_cfg_str

    if not base_path_for_eps_configs.exists():
        print(f"  ERROR: Base path for DP-SGD configs not found: {base_path_for_eps_configs}")
        return None

    optimizer_config_folders = [d.name for d in base_path_for_eps_configs.iterdir() if d.is_dir()]
    if not optimizer_config_folders:
        print(f"  ERROR: No optimizer config folders found in {base_path_for_eps_configs}")
        return None

    # Use find_best_optimizer_run_from_folder_list to select the best one based on the target_objective
    # This function expects the target objective to be part of the folder name if that's the selection criteria,
    # or it might have other internal logic. We assume it can parse and rank.
    # The original function took `results_dir_dp_eps_obj` which implies a specific structure.
    # We pass base_path_for_eps_configs which is up to .../model_cfg_str/
    
    # We need to parse the folder names to get the actual optim_cfgs
    parsed_optim_cfgs = []
    for folder_name in optimizer_config_folders:
        try:
            # optim_str_to_dict_for_vis might not be perfect if folder names have other info,
            # but it's a starting point.
            cfg = optim_str_to_dict_for_vis(folder_name)
            if cfg.get('optimizer') == 'dp_sgd': # Ensure it's a DP-SGD config
                parsed_optim_cfgs.append({'name': folder_name, 'cfg': cfg})
        except Exception as e:
            print(f"    Warning: Could not parse optim config folder name '{folder_name}': {e}")
            continue
    
    if not parsed_optim_cfgs:
        print(f"  ERROR: No valid DP-SGD optimizer configs parsed from folders in {base_path_for_eps_configs}")
        return None

    # Simplified selection: For now, take the first one found if find_best... is too complex to adapt quickly.
    # A more robust method would involve looking at 'best_scalar_to_optim.p' or similar
    # For now, let's assume one of the scripts for HP tuning created a `best_config_for_objective_X.p`
    # or that find_best_optimizer_run_from_folder_list can work with the list of names + target_objective_for_selection
    # The original find_best_optimizer_run_from_folder_list might expect a specific directory structure
    # where it can find validation performance.
    # Let's try a simpler approach: if there's a folder that matches known good parameters, use it.
    # This part is tricky without knowing the exact output structure of HP tuning for DP-SGD per objective.
    # For now, we'll try to find a config that looks plausible or take the first one.
    # This is a placeholder for a more robust selection mechanism.
    
    # Let's try to use find_best_optimizer_run_from_folder_list
    # It needs a list of full paths to the optimizer config directories
    full_paths_to_optim_cfgs = [base_path_for_eps_configs / f_name for f_name in optimizer_config_folders]

    # The `find_best_optimizer_run_from_folder_list` function from prospect's utils.exp_mgmt
    # expects to find a 'summary_val_results.csv' file within each of these paths to determine the best run.
    # It selects based on the objective specified in `target_objective_for_selection`.
    try:
        best_run_info = find_best_optimizer_run_from_folder_list(
            full_paths_to_optim_cfgs,
            target_objective_for_selection=target_objective_for_selection, # e.g., "extremile_loss"
            higher_is_better=False # Assuming loss, so lower is better
        )
        if best_run_info and 'folder_name' in best_run_info:
            selected_optim_cfg_name = Path(best_run_info['folder_name']).name
            selected_optim_cfg_dict = optim_str_to_dict_for_vis(selected_optim_cfg_name)
            print(f"  Selected DP-SGD optim_cfg folder: {selected_optim_cfg_name} with parsed_cfg: {selected_optim_cfg_dict}")
            return selected_optim_cfg_dict
        else:
            print(f"  WARNING: find_best_optimizer_run_from_folder_list did not return a best run for objective '{target_objective_for_selection}'. Falling back.")
    except Exception as e:
        print(f"  WARNING: Error using find_best_optimizer_run_from_folder_list: {e}. Falling back.")

    # Fallback: if find_best fails, take the first valid parsed DP-SGD config
    if parsed_optim_cfgs:
        print(f"  Fallback: Selecting first parsed DP-SGD config: {parsed_optim_cfgs[0]['name']}")
        return parsed_optim_cfgs[0]['cfg']
        
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot DPD for all objectives, comparing SGD and DP-SGD (eps=2).")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (e.g., acsincome).")
    parser.add_argument("--sensitive_feature", type=str, default="SEX", choices=["SEX", "RAC1P"], help="Sensitive feature to use.")
    parser.add_argument("--output_file", type=str, default="plots/fairness_comparison/dpd_all_objectives_sex_eps2.png", help="Path to save the output plot.")
    parser.add_argument("--hp_tuning_dir", type=str, default=str(HP_TUNING_DIR_PROSPECT), help="Path to the HP tuning results directory.")
    parser.add_argument("--plot_title", type=str, default=None, help="Custom plot title.")
    
    args = parser.parse_args()

    hp_tuning_dir_path = Path(args.hp_tuning_dir)
    output_file_path = Path(args.output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Dataset ---
    print(f"--- Loading dataset: {args.dataset} for Sensitive Feature: '{args.sensitive_feature}' ---")
    feature_map_acs = {"SEX": "Sex", "RAC1P": "Race"}
    dataset_fn_map = {"acsincome": load_acsincome_dataset_for_fairlearn}
    
    if args.dataset not in dataset_fn_map:
        print(f"Unsupported dataset: {args.dataset}")
        sys.exit(1)

    X_train, y_train, Z_train_df = dataset_fn_map[args.dataset](
        path_to_data_dir=hp_tuning_dir_path.parent / "data", # Assuming data is one level above hp_tuning_experiments
        sensitive_col=args.sensitive_feature,
        label_col="PINCP",
        include_intercept=False, # Weights will have intercept if model does
        N_train=4000, N_test=0, # Only need train for this script
        device='cpu', seed=DEFAULT_SEED, binarize_sensitive=True
    )
    
    # Binarize y_train for Fairlearn if it's not already 0/1
    # Fairlearn expects y_true and y_pred to be 0 or 1.
    # The ACSIncome labels from this loader are continuous initially.
    if y_train.dtype != torch.long and y_train.unique().numel() > 2:
        print(f"  WARNING: y_train labels are {y_train.unique()[:10]}. Attempting to binarize for Fairlearn.")
        y_train = (y_train > y_train.median()).long() # Example binarization
        print(f"  INFO: y_train binarized to 0/1. New unique labels: {torch.unique(y_train)}")

    y_train_np = y_train.numpy()
    sensitive_features_train_values = Z_train_df[args.sensitive_feature].values
    print(f"Dataset loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, Z_train ('{args.sensitive_feature}') shape: {sensitive_features_train_values.shape}")


    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for objectives (can be extended)
    objective_colors = {
        "erm": "blue",
        "extremile": "green",
        "esrm": "red",
        "superquantile": "purple"
    }

    all_plot_handles = []
    all_plot_labels = []

    # --- Main Loop for Objectives and Methods ---
    for objective_name in OBJECTIVES_TO_PLOT:
        print(f"\n=== Processing Objective: {objective_name} ===")
        current_model_cfg = MODEL_CFG_SHARED.copy()
        current_model_cfg['objective'] = objective_name
        
        color_base = objective_colors.get(objective_name, "gray") # Default color if objective not in map

        # --- 1. SGD ---
        print(f"  -- Method: SGD --")
        metrics_df_sgd = get_iterates_data(
            "sgd", OPTIM_CFG_SGD, current_model_cfg, args.dataset, hp_tuning_dir_path, DEFAULT_SEED
        )
        if metrics_df_sgd is not None and "iterations" in metrics_df_sgd.columns:
            sgd_dpd_scores = []
            epochs_sgd = []
            for epoch, weights_tensor in enumerate(metrics_df_sgd['iterations']):
                if not isinstance(weights_tensor, torch.Tensor):
                    # print(f"    E{epoch}: SGD W not tensor. Skip.")
                    continue
                if X_train.shape[1] != weights_tensor.shape[0]:
                    # print(f"    E{epoch}: SGD Shape mismatch. Skip.")
                    continue
                try:
                    logits_sgd = X_train @ weights_tensor.double()
                    y_pred_sgd_np = (logits_sgd >= 0).long().numpy()
                    dpd_val = demographic_parity_difference(y_train_np, y_pred_sgd_np, sensitive_features=sensitive_features_train_values)
                    sgd_dpd_scores.append(dpd_val)
                    epochs_sgd.append(epoch)
                except Exception as e:
                    print(f"    E{epoch}: SGD DPD calculation error: {e}")
                    sgd_dpd_scores.append(np.nan) # Keep epoch count aligned
                    epochs_sgd.append(epoch)
            
            if sgd_dpd_scores:
                smoothed_sgd_dpd = moving_average(sgd_dpd_scores, MOVING_AVG_WINDOW)
                epochs_to_plot_sgd = epochs_sgd[:len(smoothed_sgd_dpd)] if MOVING_AVG_WINDOW > 1 else epochs_sgd
                
                line, = ax.plot(epochs_to_plot_sgd, smoothed_sgd_dpd, label=f"{objective_name} - SGD", color=color_base, linestyle='--', alpha=0.7)
                all_plot_handles.append(line)
                all_plot_labels.append(f"{objective_name} - SGD")
        else:
            print(f"  INFO: No data or 'iterations' column for SGD for objective {objective_name}.")

        # --- 2. DP-SGD (eps=2.0) ---
        print(f"  -- Method: DP-SGD (eps={EPSILON_DP_SGD_TO_PLOT}) --")
        # model_cfg for DP-SGD selection should match the objective being processed
        selected_optim_cfg_dpsgd = select_dpsgd_optimizer_config_for_epsilon(
            hp_tuning_dir_path, args.dataset, current_model_cfg, EPSILON_DP_SGD_TO_PLOT, 
            target_objective_for_selection=current_model_cfg['objective'] # Selection based on current fair objective
        )

        if selected_optim_cfg_dpsgd:
            # Ensure all necessary keys for var_to_str are present, if any were added by select_dpsgd...
            # The `optim_str_to_dict_for_vis` should provide the core keys.
            # `dataset_length` and `shift_cost` might be needed if not in parsed dict.
            if 'dataset_length' not in selected_optim_cfg_dpsgd: selected_optim_cfg_dpsgd['dataset_length'] = OPTIM_CFG_SGD['dataset_length']
            if 'shift_cost' not in selected_optim_cfg_dpsgd: selected_optim_cfg_dpsgd['shift_cost'] = MODEL_CFG_SHARED['shift_cost']
            if 'optimizer' not in selected_optim_cfg_dpsgd : selected_optim_cfg_dpsgd['optimizer'] = 'dp_sgd'


            metrics_df_dpsgd = get_iterates_data(
                "dp_sgd", selected_optim_cfg_dpsgd, current_model_cfg, args.dataset, 
                hp_tuning_dir_path, DEFAULT_SEED, epsilon_val=EPSILON_DP_SGD_TO_PLOT
            )
            if metrics_df_dpsgd is not None and "iterations" in metrics_df_dpsgd.columns:
                dpsgd_dpd_scores = []
                epochs_dpsgd = []
                for epoch, weights_tensor in enumerate(metrics_df_dpsgd['iterations']):
                    if not isinstance(weights_tensor, torch.Tensor):
                        # print(f"    E{epoch}: DP-SGD W not tensor. Skip.")
                        continue
                    if X_train.shape[1] != weights_tensor.shape[0]:
                        # print(f"    E{epoch}: DP-SGD Shape mismatch. Skip.")
                        continue
                    try:
                        logits_dpsgd = X_train @ weights_tensor.double()
                        y_pred_dpsgd_np = (logits_dpsgd >= 0).long().numpy()
                        dpd_val = demographic_parity_difference(y_train_np, y_pred_dpsgd_np, sensitive_features=sensitive_features_train_values)
                        dpsgd_dpd_scores.append(dpd_val)
                        epochs_dpsgd.append(epoch)
                    except Exception as e:
                        print(f"    E{epoch}: DP-SGD DPD calculation error: {e}")
                        dpsgd_dpd_scores.append(np.nan)
                        epochs_dpsgd.append(epoch)

                if dpsgd_dpd_scores:
                    smoothed_dpsgd_dpd = moving_average(dpsgd_dpd_scores, MOVING_AVG_WINDOW)
                    epochs_to_plot_dpsgd = epochs_dpsgd[:len(smoothed_dpsgd_dpd)] if MOVING_AVG_WINDOW > 1 else epochs_dpsgd
                    
                    line, = ax.plot(epochs_to_plot_dpsgd, smoothed_dpsgd_dpd, label=f"{objective_name} - DP-SGD ($\\epsilon$={EPSILON_DP_SGD_TO_PLOT})", color=color_base, linestyle='-', alpha=1.0, linewidth=2)
                    all_plot_handles.append(line)
                    all_plot_labels.append(f"{objective_name} - DP-SGD ($\\epsilon$={EPSILON_DP_SGD_TO_PLOT})")
            else:
                print(f"  INFO: No data or 'iterations' column for DP-SGD (eps={EPSILON_DP_SGD_TO_PLOT}) for objective {objective_name}.")
        else:
            print(f"  INFO: No suitable DP-SGD (eps={EPSILON_DP_SGD_TO_PLOT}) optimizer config found for objective {objective_name}.")

    # --- Finalize Plot ---
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Demographic Parity Difference ({feature_map_acs.get(args.sensitive_feature, args.sensitive_feature)})")
    
    plot_title = args.plot_title if args.plot_title else \
        f"DPD Comparison (SGD vs DP-SGD $\\epsilon$={EPSILON_DP_SGD_TO_PLOT})\n" + \
        f"Dataset: {args.dataset}, Sensitive Feature: {feature_map_acs.get(args.sensitive_feature, args.sensitive_feature)}"
    ax.set_title(plot_title)
    
    if all_plot_handles:
        # Sort legend items for consistency if desired, or use handler_map for complex legends
        # ax.legend(handles=all_plot_handles, labels=all_plot_labels, loc='best', ncol=2)
        ax.legend(loc='best', ncol=max(1, len(OBJECTIVES_TO_PLOT) // 4)) # Adjust ncol based on number of objectives
    else:
        ax.text(0.5, 0.5, "No data to plot.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_file_path)
    print(f"\nPlot saved to: {output_file_path}")
    # plt.show() # Uncomment to display plot interactively


if __name__ == "__main__":
    main() 