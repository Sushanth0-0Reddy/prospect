import os
import pickle
import pandas as pd

def find_and_check_metrics_length(base_dir="hp_tuning_experiments"):
    print(f"Searching in base directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        # Exclude lbfgs results
        if "results_lbfgs" in root:
            continue

        for file in files:
            if file == "seed_1.p": # Assuming we're checking for seed 1, can be changed to seed_*.p logic if needed
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict) and "metrics" in data:
                        metrics = data["metrics"]
                        if isinstance(metrics, pd.DataFrame):
                            # Length of metrics DataFrame = n_epochs + 1 (for initial state at epoch -1)
                            num_recorded_epochs = len(metrics)
                            print(f"File: {file_path}, Recorded Epochs (Metrics Length): {num_recorded_epochs}")
                        else:
                            print(f"File: {file_path}, 'metrics' key found but not a DataFrame.")
                    elif isinstance(data, int) and data == -1: # FAIL_CODE
                         print(f"File: {file_path}, Contained FAIL_CODE (-1), indicating a diverged run.")
                    else:
                        print(f"File: {file_path}, Loaded but no 'metrics' key found or unexpected structure.")
                        
                except pickle.UnpicklingError:
                    print(f"File: {file_path}, Error: Could not unpickle.")
                except Exception as e:
                    print(f"File: {file_path}, Error: {e}")

if __name__ == "__main__":
    # You can change 'hp_tuning_experiments' to a more specific path if needed,
    # e.g., "hp_tuning_experiments/results_dp" or "hp_tuning_experiments/results_sgd"
    find_and_check_metrics_length(base_dir="hp_tuning_experiments")
    # Example for specific subdirectories:
    # find_and_check_metrics_length(base_dir="hp_tuning_experiments/results_dp/eps_2.000000/acsincome")
    # find_and_check_metrics_length(base_dir="hp_tuning_experiments/results_sgd/acsincome") 