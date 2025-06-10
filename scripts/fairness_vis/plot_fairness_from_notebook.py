import sys
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch # Assuming iterates might be torch tensors
import re # Added for parsing folder names
import argparse # Added argparse

# Fairlearn metrics
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

# Add project root to sys.path to allow importing from src
# Assumes script is in prospect/scripts/fairness_vis/
project_root_for_src = str(Path(__file__).resolve().parent.parent.parent)
if project_root_for_src not in sys.path:
    sys.path.insert(0, project_root_for_src)

try:
    from src.utils.io import var_to_str
    from src.utils.data import load_dataset
except ImportError as e:
    print(f"ERROR: Could not import a required module: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly, and src/utils modules are available.")
    sys.exit(1)

# --- Configuration Constants (Defaults, can be overridden by args) ---
PROJECT_ROOT = Path(project_root_for_src)
HP_TUNING_DIR = PROJECT_ROOT / "hp_tuning_experiments"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# These will be set by args, defaults provided in parser
# DATASET_NAME_GLOBAL = "acsincome"
# OBJECTIVE_GLOBAL = "extremile"
# SENSITIVE_FEATURE_GLOBAL = "SEX"

# These remain global as they are intrinsic to the experiment structure for acsincome
L2_REG_GLOBAL = 1.0
SHIFT_COST_GLOBAL = 1.0
LOSS_TYPE_GLOBAL = "squared_error" # For acsincome regression or binary equivalent
N_CLASS_GLOBAL = None             # For regression or binary classification

DEFAULT_SEED = 1

PLOT_CFGS_STYLES = {
    "SGD": {"label": "SGD", "color": "black", "linestyle": "-", "marker": "."},
    2.0: {"label": "DP-SGD eps=2.0", "color": "gold", "linestyle": "--", "marker": "^"},
    4.0: {"label": "DP-SGD eps=4.0", "color": "red", "linestyle": "--", "marker": "s"},
    10.0: {"label": "DP-SGD eps=10.0", "color": "blue", "linestyle": "--", "marker": "x"},
    "DP_SGD_DEFAULT": {"label": "DP-SGD eps={:.2f}", "color": "gray", "linestyle": ":", "marker": "+"}
}

MOVING_AVG_WINDOW = 5

# --- Helper function to calculate moving average ---
def moving_average(data, window_size):
    if not isinstance(data, (list, np.ndarray)) or not data: # Check if data is list or numpy array and not empty
        return np.array([]) 
    if window_size <= 0: # Invalid window size
        return np.array([])
    
    # Ensure data is a numpy array for processing
    np_data = np.array(data)
    if np_data.size == 0: # Handles case where input list might have been non-empty but resulted in empty array (e.g., list of Nones, though less likely here)
        return np.array([])

    # If data is shorter than window_size, or window_size is 1, return original data (or a copy)
    if len(np_data) < window_size or window_size == 1:
        return np_data # np_data is already a copy if 'data' was a list
    
    # 'valid' mode ensures that the convolution is only computed where the signals overlap completely.
    # The result is shorter than the input array by window_size - 1.
    averaged_data = np.convolve(np_data, np.ones(window_size)/window_size, mode='valid')
    return averaged_data

# --- Helper function to parse optimizer config folder names ---
def parse_optim_cfg_foldername(foldername: str) -> dict | None:
    cfg = {}
    # Define patterns to extract hyperparameters
    # These patterns assume the structure seen in var_to_str output
    patterns = {
        'lr': r"lr_([\d.eE+-]+)",
        'batch_size': r"batch_size_(\d+)",
        'clip_threshold': r"clip_threshold_([\d.eE+-]+)",
        'noise_multiplier': r"noise_multiplier_([\d.eE+-]+)",
        'dataset_length': r"dataset_length_(\d+)", # Often fixed, but good to parse
        'shift_cost': r"shift_cost_([\d.eE+-]+)"   # Often fixed
    }
    
    all_keys_found = True
    for key, pattern in patterns.items():
        match = re.search(pattern, foldername)
        if match:
            value_str = match.group(1)
            try:
                if key in ['batch_size', 'dataset_length']: # Integer values
                    cfg[key] = int(value_str)
                else: # Float values
                    cfg[key] = float(value_str)
            except ValueError:
                print(f"Warning: Could not parse value '{value_str}' for key '{key}' in folder '{foldername}'")
                return None # Essential parse failed
        else:
            # For this targeted parsing, assume these keys should usually be present if the folder matches a DP-SGD run.
            # However, dataset_length and shift_cost might be part of model_cfg sometimes or fixed.
            # For now, let's be strict for the main hyperparams.
            if key in ['lr', 'batch_size', 'clip_threshold', 'noise_multiplier']:
                print(f"Warning: Essential hyperparameter '{key}' not found in optim folder name: {foldername}")
                all_keys_found = False # Mark as not fully parsed if essential is missing
                # Do not return None immediately, allow partial parse for debugging if needed,
                # but ensure 'optimizer' is added and then check for essentials.

    if not all_keys_found or not all(k in cfg for k in ['lr', 'batch_size', 'clip_threshold', 'noise_multiplier']):
        # print(f"Debug: Not all essential keys found or parsed for folder: {foldername}. Parsed: {cfg}")
        # This might happen if patterns were too strict or defaults not set for all.
        # For now, if essential hyperparams (lr, bs, clip, nm) are there, proceed.
        pass

    # Add fixed/assumed values for DP-SGD from this context
    cfg['optimizer'] = 'dp_sgd' 
    # If dataset_length or shift_cost were not found by regex and are needed, add defaults
    if 'dataset_length' not in cfg:
        cfg['dataset_length'] = 4000 # Default assumption
    if 'shift_cost' not in cfg:
        cfg['shift_cost'] = 1.0 # Default assumption
        
    # Ensure all parts that var_to_str would use for dp_sgd are present
    expected_keys = ['optimizer', 'lr', 'dataset_length', 'batch_size', 'clip_threshold', 'noise_multiplier', 'shift_cost']
    if not all(k in cfg for k in expected_keys):
        # print(f"Debug: Final cfg missing some expected keys for var_to_str consistency. Folder: {foldername}, Cfg: {cfg}")
        # This might happen if patterns were too strict or defaults not set for all.
        # For now, if essential hyperparams (lr, bs, clip, nm) are there, proceed.
        pass

    return cfg

# --- Helper function to load iterates from hp_tuning_experiments ---
def get_iterates_hp_tuning(optimizer_type, optim_cfg_dict, model_cfg_dict, 
                           dataset_name_arg, seed, epsilon_val=None):
    """
    Loads iterates (seed_{seed}.p) from the hp_tuning_experiments directory.
    """
    print(f"Attempting to load iterates for: optimizer='{optimizer_type}', seed={seed}, epsilon={epsilon_val}, dataset='{dataset_name_arg}'")
    print(f"  Model Config for path: {model_cfg_dict}")
    # Always print optim_cfg_dict now, as SGD will also use it for path generation
    print(f"  Optim Config for path: {optim_cfg_dict}")

    model_cfg_str_to_use = None
    optim_cfg_str_to_use = None

    try:
        # Common for both SGD and DP-SGD now
        model_cfg_str_to_use = var_to_str(model_cfg_dict)
        optim_cfg_str_to_use = var_to_str(optim_cfg_dict)
            
    except Exception as e:
        print(f"ERROR: Failed during var_to_str conversion: {e}")
        return None
        
    print(f"  Generated model_cfg_str for path: {model_cfg_str_to_use}")
    print(f"  Generated optim_cfg_str for path: {optim_cfg_str_to_use}")

    iterates_file_path = None

    if optimizer_type.lower() == "dp_sgd":
        if epsilon_val is None:
            print("ERROR: epsilon_val must be provided for DP-SGD.")
            return None
        eps_folder_name = f"eps_{float(epsilon_val):.6f}" 
        iterates_file_path = HP_TUNING_DIR / "results_dp" / eps_folder_name / dataset_name_arg / model_cfg_str_to_use / optim_cfg_str_to_use / f"seed_{seed}.p"
    elif optimizer_type.lower() == "sgd":
        # SGD now uses a path structure similar to DP-SGD, but under "results_sgd"
        # and without an epsilon folder. It uses the provided optim_cfg_dict.
        print(f"  INFO: For SGD, attempting to load 'seed_{seed}.p' using model_cfg & optim_cfg path.")
        iterates_file_path = HP_TUNING_DIR / "results_sgd" / dataset_name_arg / model_cfg_str_to_use / optim_cfg_str_to_use / f"seed_{seed}.p"
    else:
        print(f"ERROR: Unsupported optimizer_type: {optimizer_type}")
        return None

    print(f"  Constructed iterates_file_path: {iterates_file_path}")

    if not iterates_file_path.exists():
        print(f"ERROR: Iterates file not found at: {iterates_file_path}")
        
        # Attempt to list parent directory contents for debugging if file not found
        parent_dir_to_check = iterates_file_path.parent
        if parent_dir_to_check.exists():
            print(f"  Contents of parent directory ({parent_dir_to_check}):")
            for item in parent_dir_to_check.iterdir():
                print(f"    - {item.name}")
        else:
            print(f"  Parent directory ({parent_dir_to_check}) also does not exist.")
        
        grandparent_dir_to_check = parent_dir_to_check.parent
        if grandparent_dir_to_check.exists():
            print(f"  Contents of grandparent directory ({grandparent_dir_to_check}):")
            for item in grandparent_dir_to_check.iterdir():
                print(f"    - {item.name}")
        else:
            print(f"  Grandparent directory ({grandparent_dir_to_check}) also does not exist.")

        return None

    try:
        with open(iterates_file_path, "rb") as f:
            iterates_data = pickle.load(f)
        print(f"  Successfully loaded iterates from: {iterates_file_path}")
        
        if not isinstance(iterates_data, list) or not iterates_data:
            print(f"WARNING: Iterates data loaded is not a non-empty list. Type: {type(iterates_data)}")
            # Potentially return it anyway if some downstream processing can handle it, or return None
            # For now, returning it.
        return iterates_data
    except Exception as e:
        print(f"ERROR: Failed to load or unpickle iterates from {iterates_file_path}: {e}")
        return None

# --- Main plotting function, parameterized --- 
def generate_plots_for_feature(dataset_name_arg: str, 
                               sensitive_feature_to_plot: str, 
                               objective_to_plot: str,
                               output_dir_path: Path):
    print(f"\n\n=== Generating plots for Dataset: '{dataset_name_arg}', Sensitive Feature: '{sensitive_feature_to_plot}', Objective: '{objective_to_plot}' ===")
    print(f"--- Output will be saved to: {output_dir_path} ---")
    os.makedirs(output_dir_path, exist_ok=True)

    display_sf_name = sensitive_feature_to_plot
    if sensitive_feature_to_plot == "RAC1P":
        display_sf_name = "Race"
    elif sensitive_feature_to_plot == "SEX":
        display_sf_name = "Sex"

    sgd_dpd_scores_raw = []
    sgd_dpr_scores_raw = []
    all_dpsgd_dpd_scores_raw = {}
    all_dpsgd_dpr_scores_raw = {}

    print(f"\n--- Loading dataset: {dataset_name_arg} for Sensitive Feature: '{display_sf_name}' ---") 
    try:
        X_train, y_train, _, _ = load_dataset(
            dataset=dataset_name_arg,
            data_path=str(DEFAULT_DATA_DIR) 
        )
        z_train_path = DEFAULT_DATA_DIR / dataset_name_arg / "metadata_tr.csv"
        if not z_train_path.exists():
            raise FileNotFoundError(f"Sensitive features file metadata_tr.csv not found at {z_train_path}.")
        Z_full_train = pd.read_csv(z_train_path)
        
        actual_sf_column_name = sensitive_feature_to_plot
        if sensitive_feature_to_plot not in Z_full_train.columns:
            alternatives = {"SEX": ["gender"], "RAC1P": ["race", "raceeth"]}
            found_alternative = False
            if sensitive_feature_to_plot in alternatives:
                for alt_name in alternatives[sensitive_feature_to_plot]:
                    if alt_name in Z_full_train.columns:
                        print(f"Warning: Sensitive feature column '{sensitive_feature_to_plot}' not found. Using alternative '{alt_name}'.")
                        actual_sf_column_name = alt_name
                        found_alternative = True
                        break
            if not found_alternative:
                raise ValueError(f"Sensitive feature '{sensitive_feature_to_plot}' (and known alternatives like {alternatives.get(sensitive_feature_to_plot)}) not found in Z_full_train columns: {Z_full_train.columns.tolist()}")

        sensitive_features_train_values = Z_full_train[actual_sf_column_name].values

        print(f"Dataset loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, Z_train ('{actual_sf_column_name}') shape: {sensitive_features_train_values.shape}")

        unique_labels = torch.unique(y_train)
        if not ( (len(unique_labels) == 2 and (0. in unique_labels and 1. in unique_labels)) or \
                 (len(unique_labels) == 2 and (-1. in unique_labels and 1. in unique_labels)) ):
            print(f"  WARNING: y_train labels are {unique_labels}. Attempting to binarize for Fairlearn.")
            min_val, max_val = torch.min(unique_labels), torch.max(unique_labels)
            if min_val == max_val: 
                 print(f"  ERROR: y_train contains only one label value ({min_val}). Cannot binarize for fairness metrics.")
                 return 
            else:
                y_train = (y_train == max_val).long()
                print(f"  INFO: y_train binarized to 0/1. New unique labels: {torch.unique(y_train)}")
        elif -1. in unique_labels and 1. in unique_labels and 0. not in unique_labels:
             y_train = ((y_train + 1) / 2).long()
             print(f"  INFO: y_train (-1/1) mapped to 0/1. New unique labels: {torch.unique(y_train)}")
        y_train_np = y_train.numpy()

    except Exception as e:
        print(f"ERROR: Failed to load or preprocess dataset for '{sensitive_feature_to_plot}': {e}")
        import traceback
        traceback.print_exc()
        return 
    
    model_cfg_dict = {
        "objective": objective_to_plot,
        "l2_reg": L2_REG_GLOBAL,
        "shift_cost": SHIFT_COST_GLOBAL,
        "loss": LOSS_TYPE_GLOBAL,
        "n_class": N_CLASS_GLOBAL 
    }

    # --- SGD Test Case ---
    print(f"\n--- Processing SGD Test Case for Objective: {objective_to_plot} ---")
    # Fixed SGD config based on notebook's successful runs for acsincome
    # (If SGD HPs vary per objective, this needs to become dynamic)
    optim_cfg_sgd = {
        'optimizer': 'sgd', 'lr': 0.003, 'dataset_length': 4000,
        'batch_size': 128, 'shift_cost': SHIFT_COST_GLOBAL,
        'noise_multiplier': 0.0 # Added to match path naming convention seen in logs
    }
    
    # Use dataset_name_arg for loading iterates
    sgd_loaded_data = get_iterates_hp_tuning("sgd", optim_cfg_sgd, model_cfg_dict, dataset_name_arg, DEFAULT_SEED)

    if sgd_loaded_data is not None: # Check if anything was loaded successfully
        metrics_df_sgd = None
        # Expect SGD data to be a dict like DP-SGD
        if isinstance(sgd_loaded_data, dict) and "metrics" in sgd_loaded_data and isinstance(sgd_loaded_data["metrics"], pd.DataFrame):
            print("  INFO: SGD loaded data is a dict, using 'metrics' key for DataFrame.")
            metrics_df_sgd = sgd_loaded_data["metrics"]
        # Fallback or warning if it's a direct DataFrame (less expected now, but could be a transitionary state)
        elif isinstance(sgd_loaded_data, pd.DataFrame):
            print("  WARNING: SGD loaded data is a direct DataFrame. Attempting to use, but 'iterations' column might be missing if it's from old best_traj.p.")
            metrics_df_sgd = sgd_loaded_data
        else:
            print(f"  WARNING: SGD data loaded but is not a dict with a 'metrics' DataFrame or a direct DataFrame. Type: {type(sgd_loaded_data)}")

        if metrics_df_sgd is not None: # If we successfully extracted/identified the metrics DataFrame
            print(f"  Metrics DataFrame found for SGD. Shape: {metrics_df_sgd.shape}. Columns: {metrics_df_sgd.columns.tolist()}")
            if "iterations" in metrics_df_sgd.columns:
                print(f"  Processing SGD weights from 'metrics[\'iterations\']' for DPD and DPRatio...")
                for epoch, weights_tensor in enumerate(metrics_df_sgd['iterations']):
                    if not isinstance(weights_tensor, torch.Tensor): 
                        print(f"    Epoch {epoch}: SGD Weights not tensor (type: {type(weights_tensor)}). Skip.")
                        continue
                    if X_train.shape[1] != weights_tensor.shape[0]:
                        print(f"    Epoch {epoch}: SGD Shape mismatch. X_train: {X_train.shape}, Weights: {weights_tensor.shape}. Skip.")
                        continue
                    try:
                        logits = X_train @ weights_tensor.double() 
                        y_pred_sgd_np = (logits >= 0).long().numpy()
                        
                        dpd_val = demographic_parity_difference(
                            y_train_np, y_pred_sgd_np, sensitive_features=sensitive_features_train_values
                        )
                        dpr_val = demographic_parity_ratio(
                            y_train_np, y_pred_sgd_np, sensitive_features=sensitive_features_train_values
                        )
                        sgd_dpd_scores_raw.append(dpd_val)
                        sgd_dpr_scores_raw.append(dpr_val)
                        if epoch % 10 == 0: 
                             print(f"    Epoch {epoch}: SGD DPD ({display_sf_name}) = {dpd_val:.4f}, DPR = {dpr_val:.4f}")
                    except Exception as e:
                        print(f"    Epoch {epoch}: Error SGD fairness calculation: {e}")
                if sgd_dpd_scores_raw:
                    print(f"  SGD Final DPD: {sgd_dpd_scores_raw[-1]:.4f}, Final DPR: {sgd_dpr_scores_raw[-1]:.4f} after {len(sgd_dpd_scores_raw)} epochs.")
                else: 
                    print("  No SGD fairness scores calculated.")
            else: 
                print("  'iterations' column not found in SGD metrics DataFrame.")
        else:
            print("  Failed to identify/extract metrics DataFrame from loaded SGD data.")
    else: 
        print("Failed to load SGD data.")

    # --- DP-SGD Test Cases (Iterate over found epsilon folders) ---
    print(f"\n--- Processing DP-SGD Test Cases for Objective: {objective_to_plot} ---")
    
    # Construct path to find epsilon folders for the current objective
    # Path: HP_TUNING_DIR / results_dp / eps_* / DATASET_NAME_GLOBAL / model_cfg_str_for_objective /
    model_cfg_str_for_objective = var_to_str(model_cfg_dict) # model_cfg includes the current objective
    # Use dataset_name_arg
    base_dp_path_for_objective = HP_TUNING_DIR / "results_dp" 
    
    # We need to find epsilon subfolders first, then DATASET, then MODEL_CFG_STR
    # So, the structure to search is: HP_TUNING_DIR / "results_dp" / "eps_X.XXXXXX" / DATASET_NAME_ARG / model_cfg_str_for_objective
    
    found_eps_folders_for_current_objective = []
    if base_dp_path_for_objective.exists():
        for eps_dir_candidate in base_dp_path_for_objective.iterdir():
            if eps_dir_candidate.is_dir() and eps_dir_candidate.name.startswith("eps_"):
                # Check if this epsilon folder contains the specific dataset and model_cfg for the current objective
                potential_path = eps_dir_candidate / dataset_name_arg / model_cfg_str_for_objective
                if potential_path.exists() and potential_path.is_dir():
                    found_eps_folders_for_current_objective.append(eps_dir_candidate)
    
    if not found_eps_folders_for_current_objective:
        print(f"WARNING: No DP-SGD epsilon folders found under {base_dp_path_for_objective} for dataset '{dataset_name_arg}' and model config '{model_cfg_str_for_objective}'. Skipping DP-SGD cases.")
    else:
        print(f"Found {len(found_eps_folders_for_current_objective)} epsilon folders for {dataset_name_arg}/{objective_to_plot}: {[f.name for f in found_eps_folders_for_current_objective]}")

    for eps_dir_path in found_eps_folders_for_current_objective: # eps_dir_path is like .../results_dp/eps_2.000000
        try:
            epsilon_val_str = eps_dir_path.name.replace("eps_", "")
            epsilon_val = float(epsilon_val_str)
            print(f"\n-- Processing for Epsilon: {epsilon_val} --")
        except ValueError:
            print(f"Could not parse epsilon from folder name: {eps_dir_path.name}. Skipping.")
            continue

        # Path to optimizer config folders for this epsilon, dataset, and model_cfg
        # HP_TUNING_DIR / results_dp / eps_X.XXXXXX / DATASET_NAME_ARG / model_cfg_str_for_objective / <optim_cfg_folders>
        current_experiment_base_path = eps_dir_path / dataset_name_arg / model_cfg_str_for_objective
        
        if not current_experiment_base_path.exists() or not current_experiment_base_path.is_dir():
            print(f"ERROR: Experiment path {current_experiment_base_path} does not exist or is not a directory. Should not happen if found_eps_folders logic is correct.")
            continue

        # Find the best DP-SGD run within this epsilon's directory structure based on some criteria
        # For simplicity, let's try to find one that matches a common/expected pattern if analyze_hp_results hasn't placed a 'best_cfg.p' or 'best_traj.p'
        # The original script had a fixed DP-SGD config. We need a strategy if there are multiple.
        # Option 1: Look for 'best_cfg.p' from analyze_hp_results.py.
        # Option 2: Iterate all optim_cfgs and pick one (e.g., first, or one with most epochs if seed.p contains that).
        # Option 3: Use a fixed known-good config if available (less flexible).

        # Let's try to find a suitable optim_cfg folder. For now, take the first one that seems valid.
        # This part might need refinement if there are many optim_cfg folders per epsilon.
        # A more robust approach would use outputs from analyze_hp_results.py (e.g., best_cfg.p)
        # to identify the specific optim_cfg folder.
        
        suitable_optim_cfg_dpsgd = None
        optim_cfg_folder_to_use = None

        for optim_cfg_dir in current_experiment_base_path.iterdir():
            if optim_cfg_dir.is_dir():
                # Try to parse the folder name into an optim_cfg_dict
                parsed_cfg = parse_optim_cfg_foldername(optim_cfg_dir.name)
                if parsed_cfg: # If parsing is successful and returns a dict
                    # Check if seed file exists
                    if (optim_cfg_dir / f"seed_{DEFAULT_SEED}.p").exists():
                        suitable_optim_cfg_dpsgd = parsed_cfg
                        optim_cfg_folder_to_use = optim_cfg_dir # Keep path for debug
                        print(f"  Selected optim_cfg folder: {optim_cfg_dir.name} with parsed_cfg: {suitable_optim_cfg_dpsgd}")
                        break # Use the first suitable one found
        
        if not suitable_optim_cfg_dpsgd:
            print(f"WARNING: No suitable DP-SGD optimizer config folder found under {current_experiment_base_path} with a seed_{DEFAULT_SEED}.p file and parsable name. Skipping epsilon {epsilon_val}.")
            continue
            
        # Use dataset_name_arg for loading iterates
        dpsgd_data_dict = get_iterates_hp_tuning("dp_sgd", suitable_optim_cfg_dpsgd, model_cfg_dict, dataset_name_arg, DEFAULT_SEED, epsilon_val=epsilon_val)

        if dpsgd_data_dict and isinstance(dpsgd_data_dict, dict):
            if "metrics" in dpsgd_data_dict and isinstance(dpsgd_data_dict["metrics"], pd.DataFrame):
                metrics_df_dp = dpsgd_data_dict["metrics"]
                if "iterations" in metrics_df_dp.columns:
                    print(f"      Processing DP-SGD (eps={epsilon_val:.2f}) from 'metrics[\'iterations\']'...")
                    current_eps_dpd_scores = []
                    current_eps_dpr_scores = []
                    for epoch, weights_tensor in enumerate(metrics_df_dp['iterations']):
                        if not isinstance(weights_tensor, torch.Tensor): 
                            print(f"        E{epoch}: DP-SGD W not tensor (eps={epsilon_val:.2f}). Skip.")
                            continue
                        if X_train.shape[1] != weights_tensor.shape[0]:
                            print(f"        E{epoch}: DP-SGD Shape mismatch (eps={epsilon_val:.2f}). Skip.")
                            continue
                        try:
                            logits_dp = X_train @ weights_tensor.double()
                            y_pred_dpsgd_np = (logits_dp >= 0).long().numpy()
                            dpd_val = demographic_parity_difference(y_train_np, y_pred_dpsgd_np, sensitive_features=sensitive_features_train_values)
                            dpr_val = demographic_parity_ratio(y_train_np, y_pred_dpsgd_np, sensitive_features=sensitive_features_train_values)
                            current_eps_dpd_scores.append(dpd_val)
                            current_eps_dpr_scores.append(dpr_val)
                            if epoch % 10 == 0: 
                                print(f"        E{epoch}: DP-SGD DPD(eps={epsilon_val:.2f}, {display_sf_name})={dpd_val:.4f}, DPR={dpr_val:.4f}")
                        except Exception as e:
                            print(f"        E{epoch}: Error DP-SGD fairness calc (eps={epsilon_val:.2f}): {e}")
                        
                    # Ensure these lines are OUTSIDE the for loop over epochs
                    if current_eps_dpd_scores: # Check if any scores were collected
                        all_dpsgd_dpd_scores_raw[epsilon_val] = current_eps_dpd_scores
                        all_dpsgd_dpr_scores_raw[epsilon_val] = current_eps_dpr_scores
                        print(f"      DP-SGD (eps={epsilon_val:.2f}) Final DPD: {current_eps_dpd_scores[-1]:.4f}, DPR: {current_eps_dpr_scores[-1]:.4f} after {len(current_eps_dpd_scores)} epochs.")
                    else: 
                        print(f"      No DP-SGD scores calculated for eps={epsilon_val:.2f}, config {optim_cfg_folder_to_use.name if optim_cfg_folder_to_use else 'N/A'}.")
                else: 
                    # This print should refer to optim_cfg_folder_to_use which is defined earlier
                    print(f"      'iterations' col not in DP-SGD metrics (eps={epsilon_val:.2f}, {optim_cfg_folder_to_use.name if optim_cfg_folder_to_use else 'N/A'}).")
            else: 
                # This print should refer to optim_cfg_folder_to_use
                print(f"      Metrics not found/DataFrame for DP-SGD (eps={epsilon_val:.2f}, {optim_cfg_folder_to_use.name if optim_cfg_folder_to_use else 'N/A'}).")
        else: 
            # This print should refer to optim_cfg_folder_to_use
            print(f"    Failed to load data for DP-SGD (eps={epsilon_val:.2f}, {optim_cfg_folder_to_use.name if optim_cfg_folder_to_use else 'N/A'}).")

    # --- Plotting ---
    print("\n--- Generating and Saving Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid') # Corrected style name

    # Plot for Demographic Parity Difference
    plt.figure(figsize=(12, 8))
    if sgd_dpd_scores_raw:
        sgd_dpd_smoothed = moving_average(sgd_dpd_scores_raw, MOVING_AVG_WINDOW)
        if sgd_dpd_smoothed.size > 0:
            style = PLOT_CFGS_STYLES.get("SGD", PLOT_CFGS_STYLES["DP_SGD_DEFAULT"]) # Fallback shouldn't be needed for SGD
            plt.plot(sgd_dpd_smoothed, label=style['label'], color=style['color'], 
                        linestyle=style['linestyle'], marker=style['marker'], linewidth=2)
    
    sorted_epsilons_dpd = sorted(all_dpsgd_dpd_scores_raw.keys())
    for i, epsilon_val in enumerate(sorted_epsilons_dpd):
        scores_raw = all_dpsgd_dpd_scores_raw[epsilon_val]
        if scores_raw:
            scores_smoothed = moving_average(scores_raw, MOVING_AVG_WINDOW)
            if scores_smoothed.size > 0:
                style = PLOT_CFGS_STYLES.get(epsilon_val, PLOT_CFGS_STYLES["DP_SGD_DEFAULT"])
                label = style['label'].format(epsilon_val) if "{:.2f}" in style['label'] else style['label']
                plt.plot(scores_smoothed, label=label, color=style.get('color'), 
                            linestyle=style.get('linestyle'), marker=style.get('marker'), linewidth=1.5)

    plt.title(f"Demographic Parity Difference (DPD) vs. Epoch for {display_sf_name} ({objective_to_plot.capitalize()})")
    plt.xlabel("Epoch")
    plt.ylabel(f"DPD (Smoothed, Window={MOVING_AVG_WINDOW})")
    plt.legend(loc='best')
    plt.tight_layout()
    # Use output_dir_path for saving
    dpd_plot_filename = output_dir_path / f"demographic_parity_difference_{display_sf_name}_{objective_to_plot}.png"
    plt.savefig(dpd_plot_filename)
    print(f"DPD plot saved to: {dpd_plot_filename}")
    plt.close()

    # Plot for Demographic Parity Ratio
    plt.figure(figsize=(12, 8))
    is_acs_extremile_dpr_case = dataset_name_arg == "acsincome" and objective_to_plot == "extremile"
    # For Race/extremile case, start from 5th iteration (index 4)
    start_index_dpr = 0 
    if is_acs_extremile_dpr_case and display_sf_name == "Race": # Apply only to Race + acsincome + extremile
        start_index_dpr = 4 # Start from 5th iteration (index 4)

    if sgd_dpr_scores_raw:
        data_to_smooth_sgd_dpr = sgd_dpr_scores_raw
        if start_index_dpr > 0 and len(sgd_dpr_scores_raw) > start_index_dpr:
            data_to_smooth_sgd_dpr = sgd_dpr_scores_raw[start_index_dpr:]
        elif start_index_dpr > 0: # Not enough data to start from 10th, use empty or whatever is left (likely empty) 
            data_to_smooth_sgd_dpr = sgd_dpr_scores_raw[len(sgd_dpr_scores_raw):] # Results in empty list
            
        sgd_dpr_smoothed = moving_average(data_to_smooth_sgd_dpr, MOVING_AVG_WINDOW)
        if sgd_dpr_smoothed.size > 0:
            style = PLOT_CFGS_STYLES.get("SGD")
            plt.plot(sgd_dpr_smoothed, label=style['label'], color=style['color'], 
                        linestyle=style['linestyle'], marker=style['marker'], linewidth=2)

    sorted_epsilons_dpr = sorted(all_dpsgd_dpr_scores_raw.keys())
    for i, epsilon_val in enumerate(sorted_epsilons_dpr):
        scores_raw = all_dpsgd_dpr_scores_raw[epsilon_val]
        if scores_raw:
            data_to_smooth_dpsgd_dpr = scores_raw
            if start_index_dpr > 0 and len(scores_raw) > start_index_dpr:
                data_to_smooth_dpsgd_dpr = scores_raw[start_index_dpr:]
            elif start_index_dpr > 0:
                data_to_smooth_dpsgd_dpr = scores_raw[len(scores_raw):]

            scores_smoothed = moving_average(data_to_smooth_dpsgd_dpr, MOVING_AVG_WINDOW)
            if scores_smoothed.size > 0:
                style = PLOT_CFGS_STYLES.get(epsilon_val, PLOT_CFGS_STYLES["DP_SGD_DEFAULT"])
                label = style['label'].format(epsilon_val) if "{:.2f}" in style['label'] else style['label']
                plt.plot(scores_smoothed, label=label, color=style.get('color'), 
                            linestyle=style.get('linestyle'), marker=style.get('marker'), linewidth=1.5)
    
    xlabel_dpr = f"Epoch/Iteration Step (Smoothed, Window={MOVING_AVG_WINDOW})"
    if is_acs_extremile_dpr_case and display_sf_name == "Race": # Apply only to Race + acsincome + extremile
        xlabel_dpr += " (from 5th iter.)"
    plt.title(f"Demographic Parity Ratio (DPR) vs. Epoch for {display_sf_name} ({objective_to_plot.capitalize()})")
    plt.xlabel(xlabel_dpr)
    plt.ylabel(f"DPR (Smoothed, Window={MOVING_AVG_WINDOW})")
    plt.legend(loc='best')
    plt.tight_layout()
    # Use output_dir_path for saving
    dpr_plot_filename = output_dir_path / f"demographic_parity_ratio_{display_sf_name}_{objective_to_plot}.png"
    plt.savefig(dpr_plot_filename)
    print(f"DPR plot saved to: {dpr_plot_filename}")
    plt.close()

    print(f"\n=== Finished plots for Sensitive Feature: '{sensitive_feature_to_plot}', Objective: '{objective_to_plot}' ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fairness plots from notebook logic, parameterized.")
    parser.add_argument("--dataset", type=str, default="acsincome", help="Dataset name (default: acsincome).")
    parser.add_argument("--objective", type=str, required=True, help="Objective function name (e.g., extremile, erm).")
    parser.add_argument("--sensitive_feature", type=str, default="SEX", help="Sensitive feature to analyze (default: SEX).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated plots.")
    # Potentially add --l2_reg, --shift_cost, --loss_type if they need to vary from globals

    args = parser.parse_args()

    # Convert output_dir to Path object
    output_dir = Path(args.output_dir)

    # Run the main plotting function with parsed arguments
    # The global constants L2_REG_GLOBAL, SHIFT_COST_GLOBAL, etc., will be used for model_cfg
    # unless they are also made arguments.
    generate_plots_for_feature(
        dataset_name_arg=args.dataset,
        sensitive_feature_to_plot=args.sensitive_feature,
        objective_to_plot=args.objective,
        output_dir_path=output_dir
    )

    print("\nScript finished.")

# Remove main_old_logic as its contents are now refactored

# --- main() function from before (to be replaced by generate_plots_for_feature) ---
# def main():
#    ... (all the old main function code) ...
# This will be entirely removed or refactored into generate_plots_for_feature. 