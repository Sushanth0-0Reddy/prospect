import os
import sys
import pickle
import re
import pandas as pd
from collections import defaultdict

# Ensure src can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
PARENT_VIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../hp_tuning_vis"))

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, ".."))) # for potential utility imports from parent 'scripts'
sys.path.append(SRC_DIR) # for src imports
sys.path.append(PARENT_VIS_DIR) # for importing from hp_tuning_vis

# Assuming plot_epsilon_vs_sensitivity.py (now in PARENT_VIS_DIR) contains parse_path_for_hyperparams_for_vis
from plot_epsilon_vs_sensitivity import parse_path_for_hyperparams_for_vis

FAIL_CODE = -1

def normalize_path_for_experiment_comparison(path_str):
    """Replaces clip_threshold_X.X or clip_X with a placeholder for comparing experiment sets."""
    # More specific regex to target clip_threshold or clip followed by numbers
    path_str = re.sub(r"_clip_threshold_([\d\.eE\+\-]+)", "_clip_threshold_PLACEHOLDER", path_str)
    path_str = re.sub(r"_clip_([\d\.eE\+\-]+)(?![\w\-])", "_clip_PLACEHOLDER", path_str) # Avoids matching things like _clip_norm if it existed
    return path_str

def investigate_paths_for_clip_norms(base_results_dir, dataset_name, objective_name, target_epsilon, max_paths_to_show_per_clip_norm=3):
    """
    For a fixed epsilon, investigates file paths contributing to results for each unique clip_threshold.
    Stores the path of the best run (min loss) and a sample of all paths for each clip norm.
    """
    min_loss_data_per_clip_norm = defaultdict(lambda: {'min_loss': float('inf'), 'best_path': None, 'all_paths_sample': []})
    
    search_path = os.path.join(base_results_dir, f"results_dp/eps_{target_epsilon:.6f}", dataset_name)
    print(f"Investigating DP-SGD runs (eps={target_epsilon}) in: {search_path}\n")

    paths_processed_for_clip_norm = defaultdict(list)
    normalized_paths_for_clip_norm = defaultdict(set) # Store normalized paths as a set for easy comparison

    for root, dirs, files in os.walk(search_path):
        for file_name in files:
            if file_name.startswith("seed_") and file_name.endswith(".p"):
                file_path = os.path.join(root, file_name)
                # Normalize path for consistent comparison and display
                file_path_normalized = file_path.replace("\\", "/") 

                parsed_h = parse_path_for_hyperparams_for_vis(file_path_normalized)

                if not (parsed_h["objective"] == objective_name and \
                        parsed_h["optimizer_type"] == "dp_sgd" and \
                        parsed_h["epsilon"] is not None and abs(parsed_h["epsilon"] - target_epsilon) < 1e-5 and \
                        parsed_h["clip_threshold"] is not None):
                    continue
                
                current_clip_norm = parsed_h["clip_threshold"]
                paths_processed_for_clip_norm[current_clip_norm].append(file_path_normalized)
                
                # Normalize and store for set comparison
                normalized_path = normalize_path_for_experiment_comparison(file_path_normalized)
                normalized_paths_for_clip_norm[current_clip_norm].add(normalized_path)
                
                try:
                    with open(file_path, "rb") as f: data = pickle.load(f)
                    
                    if isinstance(data, int) and data == FAIL_CODE:
                        # Still note that this path was processed for this clip norm
                        if len(min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample']) < max_paths_to_show_per_clip_norm:
                             min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample'].append(f"(FAIL_CODE) {file_path_normalized}")
                        continue
                    
                    if isinstance(data, dict) and "metrics" in data and \
                       isinstance(data["metrics"], pd.DataFrame) and not data["metrics"].empty and \
                       "train_loss" in data["metrics"].columns:
                        metrics_df = data["metrics"]
                        final_loss = metrics_df["train_loss"].iloc[-1]
                        
                        if len(min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample']) < max_paths_to_show_per_clip_norm:
                            min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample'].append(file_path_normalized)

                        if final_loss < min_loss_data_per_clip_norm[current_clip_norm]['min_loss']:
                            min_loss_data_per_clip_norm[current_clip_norm]['min_loss'] = final_loss
                            min_loss_data_per_clip_norm[current_clip_norm]['best_path'] = file_path_normalized
                except Exception as e:
                    print(f"Skipping file {file_path_normalized} due to error: {e}")
                    if len(min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample']) < max_paths_to_show_per_clip_norm:
                        min_loss_data_per_clip_norm[current_clip_norm]['all_paths_sample'].append(f"(ERROR: {e}) {file_path_normalized}")
                    continue
    
    if not min_loss_data_per_clip_norm:
        print(f"No successful runs found for epsilon {target_epsilon} to analyze.")
        return

    print(f"--- Path Investigation Results for Epsilon: {target_epsilon} ---")
    sorted_clip_norms = sorted(min_loss_data_per_clip_norm.keys())
    
    for cn in sorted_clip_norms:
        data = min_loss_data_per_clip_norm[cn]
        print(f"\nClip Norm: {cn:.2f}")
        print(f"  Min Final Loss: {data['min_loss']:.4f}")
        print(f"  Best Path: {data['best_path']}")
        print(f"  Total distinct paths processed for this clip norm: {len(paths_processed_for_clip_norm[cn])}")
        print(f"  Sample of paths processed for this clip norm (max {max_paths_to_show_per_clip_norm}):")
        for p_sample in data['all_paths_sample']:
            print(f"    - {p_sample}")
        # For more thorough check, ensure all_paths_sample is different for different clip norms if counts are high
        if len(paths_processed_for_clip_norm[cn]) > max_paths_to_show_per_clip_norm:
             print(f"    ... and {len(paths_processed_for_clip_norm[cn]) - max_paths_to_show_per_clip_norm} more paths for this clip norm.")

    # --- New: Compare sets of normalized paths between different clip norms ---
    print(f"\n--- Normalized Path Set Comparison for Epsilon: {target_epsilon} ---")
    clip_norms_list = sorted(normalized_paths_for_clip_norm.keys())
    compared_pairs = set()

    if len(clip_norms_list) < 2:
        print("Not enough clip norm categories to compare path sets.")
        return

    for i in range(len(clip_norms_list)):
        for j in range(i + 1, len(clip_norms_list)):
            cn1 = clip_norms_list[i]
            cn2 = clip_norms_list[j]
            
            set1 = normalized_paths_for_clip_norm[cn1]
            set2 = normalized_paths_for_clip_norm[cn2]

            print(f"\nComparing Clip Norm {cn1:.2f} (count: {len(set1)}) vs. Clip Norm {cn2:.2f} (count: {len(set2)}):")
            if set1 == set2:
                print(f"  Normalized experiment path sets are IDENTICAL.")
            else:
                print(f"  Normalized experiment path sets are DIFFERENT.")
                diff1_vs_2 = set1 - set2
                diff2_vs_1 = set2 - set1
                if diff1_vs_2:
                    print(f"    Paths in {cn1:.2f} but not in {cn2:.2f} (normalized, sample max 3):")
                    for p_idx, p_diff in enumerate(list(diff1_vs_2)[:3]):
                        print(f"      - {p_diff}")
                    if len(diff1_vs_2) > 3:
                        print(f"      ... and {len(diff1_vs_2) - 3} more.")
                if diff2_vs_1:
                    print(f"    Paths in {cn2:.2f} but not in {cn1:.2f} (normalized, sample max 3):")
                    for p_idx, p_diff in enumerate(list(diff2_vs_1)[:3]):
                        print(f"      - {p_diff}")
                    if len(diff2_vs_1) > 3:
                        print(f"      ... and {len(diff2_vs_1) - 3} more.")


if __name__ == "__main__":
    results_base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "../../hp_tuning_experiments"))
    dataset = "acsincome"  # Example dataset
    objective = "extremile" # Example objective
    
    # Epsilon value to investigate. You can change this.
    epsilon_to_investigate = 2.0 
    # epsilon_to_investigate = 100000.0 

    investigate_paths_for_clip_norms(results_base_dir, dataset, objective, epsilon_to_investigate, max_paths_to_show_per_clip_norm=5)

    # You can add more calls for different epsilons if needed:
    # print("\n\n===========================================================\n")
    # epsilon_to_investigate = 4.0 
    # investigate_paths_for_clip_norms(results_base_dir, dataset, objective, epsilon_to_investigate, max_paths_to_show_per_clip_norm=5) 