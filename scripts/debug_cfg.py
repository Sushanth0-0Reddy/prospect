import pickle
from pathlib import Path

def debug_config_file(cfg_path_str):
    cfg_path = Path(cfg_path_str)
    project_root = Path(__file__).resolve().parent.parent
    full_cfg_path = project_root / cfg_path

    print(f"Attempting to load: {full_cfg_path}")
    if not full_cfg_path.exists():
        print(f"Error: File not found at {full_cfg_path}")
        return

    try:
        with open(full_cfg_path, "rb") as f:
            cfg = pickle.load(f)
        print(f"Successfully loaded config file.")
        print(f"Keys in cfg: {list(cfg.keys())}")
        
        seed_value = cfg.get('seed')
        if seed_value is not None:
            print(f"Value of 'seed' key: {seed_value}")
        else:
            print(f"'seed' key not found or is None in the config.")
        
        # Print other relevant keys if they exist, like l2_reg or similar
        l2_reg_keys = ['l2_reg', 'l2_lambda', 'alpha']
        for key in l2_reg_keys:
            if key in cfg:
                print(f"Value of '{key}' key: {cfg[key]}")

    except Exception as e:
        print(f"Error processing file {full_cfg_path}: {e}")

if __name__ == "__main__":
    # Adjust the path below to the specific best_cfg.p you want to inspect
    # Path for DP-SGD with eps_10 (based on fairness_working notebook for extremile)
    path_to_check_dpsgd_eps10 = "hp_tuning_experiments/results_dp/eps_10.000000/acsincome/l2_reg_1.00e+00_loss_squared_error_objective_extremile_shift_cost_1.00e+00/dp_sgd/best_cfg.p"
    # Path for DP-SGD with eps_2 (based on user example path for seed_2.p)
    path_to_check_dpsgd_eps2 = "hp_tuning_experiments/results_dp/eps_2.000000/acsincome/l2_reg_1.00e+00_loss_squared_error_objective_extremile_shift_cost_1.00e+00/dp_sgd/best_cfg.p"
    # Path for SGD
    path_to_check_sgd = "hp_tuning_experiments/results_sgd/acsincome/l2_reg_1.00e+00_loss_squared_error_objective_extremile_shift_cost_1.00e+00/sgd/best_cfg.p"

    print("--- Checking DP-SGD (eps 10.0) config ---")
    debug_config_file(path_to_check_dpsgd_eps10)
    print("\n--- Checking DP-SGD (eps 2.0) config ---")
    debug_config_file(path_to_check_dpsgd_eps2)
    print("\n--- Checking SGD config ---")
    debug_config_file(path_to_check_sgd) 