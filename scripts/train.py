"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
import datetime
from joblib import Parallel, delayed
import sys
import argparse
import os
L2_REG = 1.0
SHIFT_COST = 1.0
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3]
BATCH_SIZES = [32, 64, 128, 256,512]
CLIP_THRESHOLDS = [0.01, 0.1,1,10,100]
# Create parser.
sys.path.append(".")
from src.utils.training import (
    OptimizationError,
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list
from src.utils.hyperparams import HYPERPARAM_LR




from dp_accounting.rdp import rdp_privacy_accountant as rdp
from dp_accounting import dp_event as event
from scipy import optimize as opt
def get_noise_multiplier(target_epsilon, target_delta, sampling, step):
    RDP_ORDERS = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )
    def objective(noise_multiplier):
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        dpevent = event.SelfComposedDpEvent(
            event.PoissonSampledDpEvent(
                sampling, event.GaussianDpEvent(noise_multiplier)
            ),
            int(step),
        )
        accountant.compose(dpevent)
        eps = accountant.get_epsilon(target_delta)
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    print("Optimal Noise",optimal_noise)
    return optimal_noise
# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "iwildcam",
        "kin8nm",
        "power",
        "acsincome",
        "diabetes",
        "amazon",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=64,
)
parser.add_argument(
    "--epoch_len",
    type=int,
    default=None,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
)
parser.add_argument(
    "--dataset_length",
    type=int,
    default=64,
)
parser.add_argument(
    "--use_hyperparam",
    type=int,
    default=0,
)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--n_jobs", type=int, default=-2)
parser.add_argument("--single_lr", type=float, default=None, help="Run with a single learning rate.")
parser.add_argument("--single_batch_size", type=int, default=None, help="Run with a single batch size.")
parser.add_argument("--single_clip_threshold", type=float, default=None, help="Run with a single clip threshold (for DP-SGD).")
parser.add_argument("--output_base_dir", type=str, default="hp_tuning_experiments", help="Base directory for output results.")

args = parser.parse_args()
print(f"DEBUG: Parsed args: {args}") # DEBUG
print(f"DEBUG: Raw args.optimizer from parser: '{args.optimizer}'") # DEBUG: ADDED THIS LINE

if not args.optimizer or args.optimizer.strip() == "":
    print(f"ERROR: Optimizer argument is missing or empty: '{args.optimizer}'")
    sys.exit(1)

if args.delta is None:
    args.delta = args.dataset_length**(-1.1)

# Configure for input to trainers.
dataset = args.dataset
if dataset in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
    loss = "squared_error"
    n_class = None
elif dataset == "iwildcam":
    loss = "multinomial_cross_entropy"
    n_class = 60
elif dataset == "amazon":
    loss = "multinomial_cross_entropy"
    n_class = 5
elif dataset == "diabetes":
    loss = "binary_cross_entropy"
    n_class = None

model_cfg = {
    "objective": args.objective,
    "l2_reg": L2_REG,
    "shift_cost": SHIFT_COST,
    "loss": loss,
    "n_class": n_class
}


if args.use_hyperparam:
    lrs = [HYPERPARAM_LR[args.optimizer][args.dataset][args.objective]]
    # If using a single hyperparameter, we might not want to iterate over batch sizes here
    # Or, we ensure HYPERPARAM_LR also contains batch_size if that's intended
    batch_sizes_to_use = [args.batch_size] 
else:
    lrs = [args.single_lr] if args.single_lr is not None else LRS
    batch_sizes_to_use = [args.single_batch_size] if args.single_batch_size is not None else BATCH_SIZES


optim_cfg = {
    "optimizer": args.optimizer,
    "lr": lrs,
    "epoch_len": args.epoch_len,
    "shift_cost": SHIFT_COST,
    "dataset_length":args.dataset_length,
    "batch_size": batch_sizes_to_use, # Use the determined batch sizes
}

if args.optimizer == "dp_sgd":
    clip_thresholds_to_use = [args.single_clip_threshold] if args.single_clip_threshold is not None else CLIP_THRESHOLDS
    optim_cfg["clip_threshold"] = clip_thresholds_to_use
print(f"DEBUG: optim_cfg before dict_to_list: {optim_cfg}") # DEBUG


# --- Seeds ---
# SEEDS = [1, 2] # Original
if args.single_lr is not None:
    SEEDS = [1]
    print(f"DEBUG: Running with single seed configuration: SEEDS = {SEEDS}") # DEBUG
else:
    SEEDS = [1, 2]
    print(f"DEBUG: Running with default seeds: SEEDS = {SEEDS}") # DEBUG


# Pre-calculate noise multipliers for DP-SGD
# Noise multiplier depends on epsilon, delta, batch_size, dataset_length, n_epochs
noise_multipliers_cache = {}
if args.optimizer == "dp_sgd" and args.epsilon is not None:
    print(f"DEBUG: Pre-calculating noise multipliers for epsilon={args.epsilon}, delta={args.delta}, n_epochs={args.n_epochs}")
    for bs in batch_sizes_to_use: 
        # Ensure dataset_length is available, it might be in optim_cfg or args
        dataset_len = args.dataset_length # optim_cfg["dataset_length"] would be a list here
        steps = (args.n_epochs * dataset_len) / bs
        cache_key = (args.epsilon, args.delta, bs, dataset_len, args.n_epochs)
        print(f"DEBUG: Calculating noise_multiplier for batch_size={bs}, steps={steps}") # DEBUG
        noise_multipliers_cache[bs] = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            sampling=bs / dataset_len,
            step=steps
        )
        print(f"DEBUG: Cached noise_multiplier for batch_size={bs}: {noise_multipliers_cache[bs]}") # DEBUG

optim_cfgs = dict_to_list(optim_cfg)

# Assign pre-calculated noise multipliers
for optim_conf in optim_cfgs:
    if args.optimizer == "dp_sgd" and args.epsilon is not None:
        # Assign the cached noise multiplier based on the batch_size in this specific optim_conf
        optim_conf["noise_multiplier"] = noise_multipliers_cache[optim_conf["batch_size"]]
        print(f"DEBUG: Assigned noise_multiplier={optim_conf['noise_multiplier']} to optim_conf for batch_size={optim_conf['batch_size']}") # DEBUG
    else:
        optim_conf["noise_multiplier"] = 0.0
        # print(f"DEBUG: Setting noise_multiplier to 0.0 for optim_conf: {optim_conf}") # DEBUG (can be verbose)

print(f"DEBUG: optim_cfgs after noise_multiplier assignment: {optim_cfgs}") # DEBUG

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim_cfg,
    "parallel": bool(args.parallel),
    "seeds": SEEDS,
    "n_epochs": args.n_epochs,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print(f"Start:" + " " * 11, {str(datetime.datetime.now())})
print("-----------------------------------------------------------------")

def create_output_path(optimizer_type, epsilon_val, base_dir_val, dataset_name):
    print(f"DEBUG: create_output_path called with optimizer_type='{optimizer_type}', epsilon_val={epsilon_val}, base_dir_val='{base_dir_val}', dataset_name='{dataset_name}'") # DEBUG
    """
    Creates a safe output path string with proper error handling.
    """
    try:
        # Ensure numeric values are valid for epsilon_val if optimizer is dp_sgd
        if optimizer_type == "dp_sgd":
            if epsilon_val is None or not isinstance(epsilon_val, (int, float)):
                print(f"DEBUG: Invalid type or missing epsilon_val for dp_sgd in create_output_path: {type(epsilon_val)}") # DEBUG
                raise ValueError("Invalid or missing epsilon_val for dp_sgd optimizer")

        if optimizer_type == "sgd":
            # base_dir_val is already "hp_tuning_experiments/results_sgd"
            path = os.path.join(base_dir_val, dataset_name)
            print(f"DEBUG: Path for SGD: {path}") # DEBUG
        elif optimizer_type == "dp_sgd":
            # base_dir_val is "hp_tuning_experiments/results_dp"
            # Epsilon has been validated above for dp_sgd
            path = os.path.join(base_dir_val, f"eps_{float(epsilon_val):.6f}", dataset_name)
            print(f"DEBUG: Path for DP (epsilon {epsilon_val}): {path}") # DEBUG
        else:
            # This case should ideally be caught before calling create_output_path
            print(f"ERROR: Unrecognized optimizer_type '{optimizer_type}' in create_output_path.")
            # Fallback to a generic path to avoid crashing here, but indicates a problem upstream
            path = os.path.join(base_dir_val, "unknown_optimizer_type", dataset_name)


        # Create directories if they don't exist
        os.makedirs(path, exist_ok=True)
        print(f"DEBUG: os.makedirs called for path: {path}") # DEBUG
        return path
    except ValueError as ve:
        print(f"Error creating output path: {str(ve)}")
        # Fallback to a safe default with timestamp
        fallback_path = os.path.join(base_dir_val, f"backup_{int(time.time())}")
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path
    except OSError as e:
        print(f"Error creating directory: {str(e)}")
        # Fallback to current directory if all else fails
        return "."

# Determine the correct sub-directory (results_sgd or results_dp) and the full base path
if args.optimizer == "sgd":
    # For SGD, results go into output_base_dir/results_sgd
    base_dir_for_path = os.path.join(args.output_base_dir, "results_sgd")
    print(f"DEBUG: base_dir_for_path set to: {base_dir_for_path} for optimizer sgd") # DEBUG
elif args.optimizer == "dp_sgd":
    if args.epsilon is None:
        print("ERROR: Epsilon value must be provided for dp_sgd optimizer via --epsilon flag.")
        sys.exit(1)
    # For DP-SGD, results go into output_base_dir/results_dp
    base_dir_for_path = os.path.join(args.output_base_dir, "results_dp")
    print(f"DEBUG: base_dir_for_path set to: {base_dir_for_path} for optimizer {args.optimizer}") # DEBUG
else:
    print(f"ERROR: Unrecognized optimizer specified: {args.optimizer}. Cannot determine base_dir_for_path.")
    sys.exit(1)

# Usage in your code:
try:
    # Pass optimizer_type and the actual args.epsilon (which can be None for sgd)
    output_path = create_output_path(
        optimizer_type=args.optimizer,
        epsilon_val=args.epsilon, 
        base_dir_val=base_dir_for_path, 
        dataset_name=args.dataset
    )
    print(f"DEBUG: output_path after create_output_path call: {output_path}") # DEBUG
except OSError as e:
    print(f"DEBUG: OSError during output_path creation or usage: {e}") # DEBUG
    print("Error: Failed to create or access the output directory.")
    sys.exit(1) # Exit if directory creation fails.
    
    

# Run optimization.
def worker(optim):
    print(f"DEBUG_WORKER: Worker received optim['optimizer']: '{optim.get('optimizer')}', full optim: {optim}") # DEBUG: ADDED THIS LINE
    name, lr = optim["optimizer"], optim["lr"]
    diverged = False
    for seed in SEEDS:
        code = compute_training_curve(
            dataset,
            model_cfg,
            optim,
            seed,
            args.n_epochs,
            out_path=output_path
        )
        if code == FAIL_CODE:
            diverged = True
    if diverged:
        print(f"Optimizer '{name}' diverged at learning rate {lr}!")


print(f"DEBUG: Final output_path before worker call: {output_path}") # DEBUG
tic = time.time()
if bool(args.parallel):
    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(worker)(optim) for optim in optim_cfgs)
else:
    for optim in optim_cfgs:
        worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")

# Save best configuration.
# Ensure find_best_optim_cfg also expects 'noise_multiplier' if it uses that field.
# For now, assuming find_best_optim_cfg uses what's in optim_cfgs.
find_best_optim_cfg(dataset, model_cfg, optim_cfgs, SEEDS,out_path=output_path)
