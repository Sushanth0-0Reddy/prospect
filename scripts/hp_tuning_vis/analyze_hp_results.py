import os
import pickle
import re
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import sys # Added for redirecting stdout
import traceback # For printing full traceback on error

FAIL_CODE = -1

def parse_path_for_hyperparams(path_str):
    """
    Parses hyperparameters from a directory path string.
    Assumes a specific directory structure like:
    .../results_dp/eps_2.000000/acsincome/l2_reg_.../batch_size_128_clip_threshold_1.0_..._lr_0.001_.../seed_1.p
    .../results_sgd/acsincome/l2_reg_.../batch_size_128_..._lr_0.001_.../seed_1.p
    """
    # Normalize path separators to forward slashes for consistent regex matching
    path_str = path_str.replace("\\", "/")

    hyperparams = {
        "lr": None,
        "batch_size": None,
        "clip_threshold": None, # Only for DP
        "epsilon": None, # Only for DP
        "optimizer_type": None,
        "dataset": None,
        "objective": None # Simplified for now, might need more robust parsing
    }

    # Optimizer type
    if "results_dp" in path_str:
        hyperparams["optimizer_type"] = "dp_sgd"
    elif "results_sgd" in path_str:
        hyperparams["optimizer_type"] = "sgd"

    # Epsilon for DP
    eps_match = re.search(r"eps_([\d\.]+)", path_str)
    if eps_match:
        try:
            hyperparams["epsilon"] = float(eps_match.group(1))
        except ValueError:
            pass # Should already be float from regex but good to be safe

    # Dataset (assuming it's after optimizer_type/epsilon)
    dataset_match = re.search(r"(results_dp/eps_[\d\.]+|results_sgd)/([^/]+)/", path_str)
    if dataset_match:
        hyperparams["dataset"] = dataset_match.group(2)
    
    # Objective parsing:
    # Remove old objective parsing logic.
    # New logic: Get model_config_folder_name and parse 'objective_(\w+)' from it.
    try:
        path_obj = Path(path_str) # path_str is the full file path to seed_X.p
        model_config_folder_name = path_obj.parent.parent.name # This should be the model_cfg_str directory
        
        objective_match_new = re.search(r"objective_(\w+)", model_config_folder_name)
        if objective_match_new:
            hyperparams["objective"] = objective_match_new.group(1)
        # If no match, hyperparams["objective"] remains None. 
        # The script later checks for this when grouping results.
    except IndexError:
        # This might happen if the path structure is unexpected (e.g., too short)
        print(f"Warning: Could not determine model_config_folder_name from path: {path_str} for objective parsing.")
        hyperparams["objective"] = None


    # Common Hyperparameters
    lr_match = re.search(r"lr_([\d\.eE\+\-]+)", path_str)
    if lr_match:
        try:
            hyperparams["lr"] = float(lr_match.group(1))
        except ValueError:
            pass

    bs_match = re.search(r"batch_size_(\d+)", path_str)
    if bs_match:
        try:
            hyperparams["batch_size"] = int(bs_match.group(1))
        except ValueError:
            pass

    # Clip Threshold (only for DP-SGD)
    if hyperparams["optimizer_type"] == "dp_sgd":
        ct_match = re.search(r"clip_threshold_([\d\.eE\+\-]+)", path_str)
        if ct_match:
            try:
                hyperparams["clip_threshold"] = float(ct_match.group(1))
            except ValueError:
                pass
    
    return hyperparams


def get_final_loss(metrics_df):
    """
    Extracts the final training loss from the metrics DataFrame.
    Assumes 'train_loss' column and the last row is the final epoch.
    """
    if not isinstance(metrics_df, pd.DataFrame) or "train_loss" not in metrics_df.columns:
        return float('inf') # Or some other indicator of missing data
    if metrics_df.empty:
        return float('inf')
    return metrics_df["train_loss"].iloc[-1]


def analyze_results(base_dir="hp_tuning_experiments"):
    output_file_path = Path(base_dir) / "hp_analysis_summary.txt"
    original_stdout = sys.stdout
    file_stream = None 

    try:
        file_stream = open(output_file_path, 'w')
        sys.stdout = file_stream 

    print(f"Analyzing results in: {base_dir}\n")

        all_runs = []
        diverged_runs_counter = 0 # Renamed to avoid conflict if 'diverged_runs' is used elsewhere
        total_runs = 0
        
    debug_print_count = 0
        max_debug_prints = 5

    for root, dirs, files in os.walk(base_dir):
        if "results_lbfgs" in root: # Skip L-BFGS
            continue

        for file_name in files:
            if file_name.startswith("seed_") and file_name.endswith(".p"):
                total_runs += 1
                file_path = os.path.join(root, file_name)
                hyperparams = parse_path_for_hyperparams(file_path)
                
                try:
                        with open(file_path, "rb") as f_pickle: # Use different var name for pickle file
                            data = pickle.load(f_pickle)
                    
                    loss = float('inf')
                    diverged = False

                    if isinstance(data, int) and data == FAIL_CODE:
                        diverged = True
                            diverged_runs_counter += 1
                    elif isinstance(data, dict) and "metrics" in data:
                        loss = get_final_loss(data["metrics"])
                        if loss == float('inf') and not data["metrics"].empty : # check if it became inf due to no train_loss col
                             print(f"Warning: Could not determine final loss for {file_path}, metrics present but no train_loss or empty.")
                        # A very high loss might also indicate practical divergence
                        # if loss > 1e5: # Arbitrary threshold for practical divergence
                        #     diverged = True 
                        #     diverged_runs +=1 # if we count this
                        if not diverged and debug_print_count < max_debug_prints:
                            print(f"DEBUG_OPTIMALITY_CHECK: Non-diverged run. Parsed Hyperparams: {hyperparams}, Final Loss: {loss}")
                            debug_print_count += 1
                    else:
                        # Could be an old format or unexpected structure
                        print(f"Warning: Unexpected data structure in {file_path}")
                        # Consider if these should count as diverged or be handled differently
                    
                    all_runs.append({
                        "path": file_path,
                        "hyperparams": hyperparams,
                        "final_loss": loss,
                        "diverged": diverged
                    })

                    except Exception as e_inner: # Catch error during individual file processing
                        print(f"Error processing file {file_path}: {e_inner}")
                    # Optionally count these as diverged or track separately
                        # diverged_runs_counter += 1 
                    all_runs.append({
                        "path": file_path,
                        "hyperparams": hyperparams,
                        "final_loss": float('inf'), # Treat as diverged if error
                        "diverged": True 
                    })
                        diverged_runs_counter +=1 # Count as diverged if pickle load or inner processing fails

    if not all_runs:
        print("No result files found to analyze.")
        else:
    print("--- Divergence Statistics ---")
            print(f"Total runs processed (files found): {total_runs}")
            print(f"Total actual .p files parsed (attempted): {len(all_runs)}") # Should match total_runs if no os.walk issues
            
            # Calculate diverged based on the 'diverged' flag in all_runs
            actual_diverged_count = sum(1 for r in all_runs if r['diverged'])
            print(f"Total diverged runs (FAIL_CODE or error during load/processing): {actual_diverged_count}")
            if total_runs > 0: # Use total_runs as the denominator for overall percentage
                print(f"Overall divergence percentage: {actual_diverged_count / total_runs * 100:.2f}%")

    # Divergence by optimizer
    divergence_by_optimizer = defaultdict(lambda: {"diverged": 0, "total": 0})
    for run in all_runs:
        opt_type = run["hyperparams"]["optimizer_type"]
        if opt_type:
            divergence_by_optimizer[opt_type]["total"] += 1
            if run["diverged"]:
                divergence_by_optimizer[opt_type]["diverged"] += 1
    
    for opt_type, counts in divergence_by_optimizer.items():
        perc = counts["diverged"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"  {opt_type}: {counts['diverged']}/{counts['total']} diverged ({perc:.2f}%)")

    # TODO: Add more detailed divergence stats (e.g., per hyperparameter combo if useful)

    print("\n--- Optimality Statistics (Lowest Loss for Non-Diverged Runs) ---")
    best_runs_criteria = defaultdict(lambda: {"best_loss": float('inf'), "best_run_info": None})

    # Group by: dataset, objective, optimizer_type, (epsilon for dp_sgd)
    for run in all_runs:
        if not run["diverged"] and run["hyperparams"]["dataset"] and run["hyperparams"]["objective"] and run["hyperparams"]["optimizer_type"]:
            h = run["hyperparams"]
            # Create a unique key for the grouping criteria
            criteria_key_tuple = [h["dataset"], h["objective"], h["optimizer_type"]]
                    if h["optimizer_type"] == "dp_sgd" and h["epsilon"] is not None: # Ensure epsilon is not None
                criteria_key_tuple.append(h["epsilon"])
            
            criteria_key = tuple(criteria_key_tuple)

            if run["final_loss"] < best_runs_criteria[criteria_key]["best_loss"]:
                best_runs_criteria[criteria_key]["best_loss"] = run["final_loss"]
                best_runs_criteria[criteria_key]["best_run_info"] = h # Store the hyperparams of the best run

    if not best_runs_criteria:
        print("No non-diverged runs found to determine optimality.")
    else:
        print(f"Found {len(best_runs_criteria)} unique (dataset, objective, optimizer_type, [epsilon]) combinations with a best run.")

        # Counters for how many times a hyperparameter value was part of a "best run"
        optimal_lr_counts = Counter()
        optimal_bs_counts = Counter()
        optimal_ct_counts = Counter() # Clip Threshold for DP

        for criteria_key, data in best_runs_criteria.items():
                    if data.get("best_run_info"): 
                best_h = data["best_run_info"]
                # Add a prefix to distinguish between SGD and DP-SGD hyperparams if they overlap
                # e.g. lr_sgd_0.01, lr_dp_sgd_0.01
                opt_prefix = best_h["optimizer_type"]
                eps_suffix = f"_eps{best_h['epsilon']}" if best_h["optimizer_type"] == 'dp_sgd' and best_h['epsilon'] is not None else ""

                if best_h["lr"] is not None:
                    optimal_lr_counts[f"lr_{opt_prefix}{eps_suffix}_{best_h['lr']}"] += 1
                if best_h["batch_size"] is not None:
                    optimal_bs_counts[f"bs_{opt_prefix}{eps_suffix}_{best_h['batch_size']}"] += 1
                if best_h["optimizer_type"] == "dp_sgd" and best_h["clip_threshold"] is not None:
                    optimal_ct_counts[f"ct_{opt_prefix}{eps_suffix}_{best_h['clip_threshold']}"] += 1
        
        total_best_configs_found = len(best_runs_criteria)

        print("\n  Percentage of time each hyperparameter value was in an OPTIMAL configuration:")
        print("  (Optimal = best loss for a given dataset, objective, optimizer, [and epsilon for DP])")

        if optimal_lr_counts:
            print("\n  Learning Rates:")
            for val, count in optimal_lr_counts.most_common():
                print(f"    {val}: {count/total_best_configs_found*100:.2f}% ({count}/{total_best_configs_found} times)")
        
        if optimal_bs_counts:
            print("\n  Batch Sizes:")
            for val, count in optimal_bs_counts.most_common():
                print(f"    {val}: {count/total_best_configs_found*100:.2f}% ({count}/{total_best_configs_found} times)")

        if optimal_ct_counts:
            print("\n  Clip Thresholds (DP-SGD only):")
            for val, count in optimal_ct_counts.most_common():
                print(f"    {val}: {count/total_best_configs_found*100:.2f}% ({count}/{total_best_configs_found} times)")

            # This section prints detailed optimal hyperparams per objective
            print("\n--- Detailed Optimal Hyperparameters per Objective ---")
            # Extract all unique objectives that have a best run
            unique_objectives = sorted(list(set(key[1] for key, val in best_runs_criteria.items() if val.get("best_run_info"))))

            if not unique_objectives:
                print("No best runs with hyperparameter information found to detail.")
            else:
                for obj_name in unique_objectives:
                    print(f"\nObjective: {obj_name}")
                    # Sort by optimizer type then epsilon for consistent output order
                    sorted_criteria_for_obj = sorted(
                        [(ck, cd) for ck, cd in best_runs_criteria.items() if ck[1] == obj_name and cd.get("best_run_info")],
                        key=lambda x: (x[0][2], x[0][3] if len(x[0]) > 3 and x[0][3] is not None else float('-inf')) 
                    )
                    
                    if not sorted_criteria_for_obj:
                        print(f"  No best runs found for objective '{obj_name}' with detailed hyperparameters.")
                        continue

                    for criteria_key, data in sorted_criteria_for_obj:
                       best_h = data["best_run_info"] 
                       dataset_name = criteria_key[0]
                       optimizer_type = criteria_key[2]
                       epsilon_val = criteria_key[3] if len(criteria_key) > 3 and optimizer_type == "dp_sgd" else None

                       print(f"\n  Configuration for Dataset: {dataset_name}, Optimizer: {optimizer_type}")
                       if optimizer_type == "dp_sgd":
                           # Use epsilon from hyperparams dict as criteria_key[3] might not always be there (e.g. if grouping changes)
                           epsilon_to_display = epsilon_val if epsilon_val is not None else "N/A" 
                           print(f"    Epsilon: {epsilon_to_display}")
                       print(f"    Optimal LR: {best_h['lr']}")
                       print(f"    Optimal Batch Size: {best_h['batch_size']}")
                       if optimizer_type == "dp_sgd":
                           print(f"    Optimal Clip Threshold: {best_h['clip_threshold'] if best_h['clip_threshold'] is not None else 'N/A'}")
                       print(f"    Best Loss Achieved: {data['best_loss']:.4f}")
                       # print(f"    Best run file: {data.get('best_run_path', 'N/A')}") # Optional: for debugging in log

        # Ensure this is the VERY LAST print to the log file before stdout is restored.
        print(f"\n\nAnalysis complete. Log saved to: {str(output_file_path.resolve())}\n")

    except Exception as e_outer:
        sys.stdout = original_stdout # Ensure stdout is restored before printing error to console
        print(f"AN UNEXPECTED ERROR OCCURRED DURING SCRIPT EXECUTION: {e_outer}")
        traceback.print_exc() # Print full traceback to console
    finally:
        sys.stdout = original_stdout
        if file_stream is not None:
            if not file_stream.closed:
                file_stream.close()

    # This message will now definitely go to the console.
    print(f"Script execution finished. Full output expected in: {str(output_file_path.resolve())}")


if __name__ == "__main__":
    analyze_results(base_dir="hp_tuning_experiments") 