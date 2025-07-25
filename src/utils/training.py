import pandas as pd
import time
from tqdm import tqdm
import torch
import pickle
import os
import datetime

from src.optim.baselines import (
    StochasticSubgradientMethod,
    StochasticRegularizedDualAveraging,
    SmoothedLSVRG,
    SaddleSAGA,
)
from src.optim.baselines_dp import StochasticSubgradientMethodDP

from src.optim.prospect import Prospect, ProspectMoreau
from src.optim.objectives import (
    Objective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)
from src.utils.io import save_results, load_results, var_to_str, get_path
from src.utils.data import load_dataset

SUCCESS_CODE = 0
FAIL_CODE = -1

class OptimizationError(RuntimeError):
    pass


def train_model(optimizer, val_objective, n_epochs):
    epoch_len = optimizer.get_epoch_len()
    metrics = [compute_metrics(-1, optimizer, val_objective, 0.0)]
    init_loss = metrics[0]["train_loss"]

    for epoch in tqdm(range(n_epochs)):
        tic = time.time()
        optimizer.start_epoch()
        for _ in range(epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        toc = time.time()

        # Logging.
        metrics.append(compute_metrics(epoch, optimizer, val_objective, toc - tic))
        if metrics[-1]["train_loss"] >= 1.5 * init_loss:
            raise OptimizationError(
                f"train loss 50% greater than inital loss! (epoch {epoch})"
            )

    result = {

        "weights": optimizer.weights,
        "metrics": pd.DataFrame(metrics),
    }
    return result


def get_optimizer(optim_cfg, objective, seed, device="cpu"):
    name, lr, epoch_len, shift_cost, batch_size = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["shift_cost"],
        optim_cfg["batch_size"],
    )

    lrd = 0.5 if "lrd" not in optim_cfg.keys() else optim_cfg["lrd"]
    penalty = "l2"

    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len,batch_size=batch_size
        )
        
    elif name == "dp_sgd":
        if "noise_multiplier" not in optim_cfg:
            raise ValueError("'noise_multiplier' not found in optim_cfg for dp_sgd")
        noise_multiplier = optim_cfg["noise_multiplier"]
        clip_threshold = optim_cfg.get("clip_threshold", 1.0)
        return StochasticSubgradientMethodDP(
            objective, lr=lr, seed=seed, epoch_len=epoch_len,batch_size=batch_size,noise_multiplier=noise_multiplier, clip_threshold=clip_threshold
        )
        
        
    elif name == "srda":
        return StochasticRegularizedDualAveraging(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "lsvrg":
        return SmoothedLSVRG(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
            batch_size=batch_size
        )
    elif name == "saddlesaga":
        # best lr for V1.
        return SaddleSAGA(
            objective,
            lrp=lr,
            lrd=lr / 10,
            smoothing=penalty,
            sm_coef=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
        )
    elif name == "prospect":
        return Prospect(
            objective,
            lrp=lr,
            epoch_len=epoch_len,
            shift_cost=shift_cost,
            penalty=penalty,
            seed_grad=seed,
            seed_table=3 * seed,
        )
    elif name == "moreau":
        return ProspectMoreau(
            objective,
            lr=lr,
            penalty=penalty,
            shift_cost=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
            device=device,
        )
    else:
        raise ValueError("Unreocgnized optimizer!")


def get_objective(model_cfg, X, y, dataset=None, autodiff=True, current_clip_threshold=None):
    name, l2_reg, loss, n_class, shift_cost = (
        model_cfg["objective"],
        model_cfg["l2_reg"],
        model_cfg["loss"],
        model_cfg["n_class"],
        model_cfg["shift_cost"],
    )
    if name == "erm":
        weight_function = lambda n: get_erm_weights(n)
    elif name == "extremile":
        weight_function = lambda n: get_extremile_weights(n, 2.0)
    elif name == "superquantile":
        weight_function = lambda n: get_superquantile_weights(n, 0.5)
    elif name == "esrm":
        weight_function = lambda n: get_esrm_weights(n, 1.0)
    elif name == "extremile_lite":
        weight_function = lambda n: get_extremile_weights(n, 1.5)
    elif name == "superquantile_lite":
        weight_function = lambda n: get_superquantile_weights(n, 0.25)
    elif name == "esrm_lite":
        weight_function = lambda n: get_esrm_weights(n, 0.5)
    elif name == "extremile_hard":
        weight_function = lambda n: get_extremile_weights(n, 2.5)
    elif name == "superquantile_hard":
        weight_function = lambda n: get_superquantile_weights(n, 0.75)
    elif name == "esrm_hard":
        weight_function = lambda n: get_esrm_weights(n, 2.0)

    # Use the provided current_clip_threshold if available, otherwise Objective's default (1.0) will apply.
    # The Objective class itself has a default of 1.0 for its clip_threshold parameter.
    # So, if current_clip_threshold is None (e.g. for non-DP objectives), it will correctly default.
    return Objective(
        X,
        y,
        weight_function,
        l2_reg=l2_reg,
        loss=loss,
        n_class=n_class,
        risk_name=name,
        dataset=dataset,
        shift_cost=shift_cost,
        penalty="l2",
        autodiff=autodiff,
        clip_threshold=current_clip_threshold if current_clip_threshold is not None else 1.0 # Explicitly pass, defaulting here for clarity
    )


def compute_metrics(epoch, optimizer, val_objective, elapsed):
    return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(
            optimizer.weights, include_reg=False
        ).item(),
        "val_loss": val_objective.get_batch_loss(optimizer.weights).item(),
        "elapsed": elapsed,
        "iterations":optimizer.weights
    }

def compute_training_curve(
    dataset,
    model_cfg,
    optim_cfg,
    seed,
    n_epochs,
    out_path="results/",
    data_path="data/"
):
    X_train, y_train, X_val, y_val = load_dataset(dataset, data_path=data_path)

    if model_cfg["loss"] == "multinomial_cross_entropy":
        model_cfg["n_class"] = len(torch.unique(y_train))

    # check if result exists
    if (
        result_exists(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
    ):
        print("*** Result exists ***")
        print("*********************")
        print(f"dataset: {dataset}")
        print(f"model_cfg: {model_cfg}")
        print(f"optim_cfg: {optim_cfg}")
        print(f"seed: {seed}")
        print("*********************")
        exit_code = SUCCESS_CODE
    else:
        # Extract clip_threshold for the current run from optim_cfg
        # This will be None if not present (e.g., for SGD optimizer)
        run_clip_threshold = optim_cfg.get("clip_threshold") 

        train_objective = get_objective(model_cfg, X_train, y_train, dataset=dataset,autodiff=False, current_clip_threshold=run_clip_threshold)
        val_objective = get_objective(model_cfg, X_val, y_val, dataset=dataset) # Val objective doesn't need clipping info
        optimizer = get_optimizer(optim_cfg, train_objective, seed)
        try:
            result = train_model(optimizer, val_objective, n_epochs)
            exit_code = SUCCESS_CODE
        except OptimizationError as e:
            result = FAIL_CODE
            exit_code = FAIL_CODE
        save_results(result, model_cfg, optim_cfg, seed, out_path=out_path)
        return exit_code


def result_exists(dataset, model_cfg, optim_cfg, seed, out_path="results"):
    path = "/".join([out_path, dataset, var_to_str(model_cfg), var_to_str(optim_cfg)])
    f = os.path.join(path, f"seed_{seed}.p")
    return os.path.exists(f)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_average_train_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    return total / len(seeds)

def find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds, out_path="results/"):
    # Compute optimal hyperparameters by lowest average final train loss.
    best_loss = torch.inf
    best_traj = None
    best_cfg = None
    for optim_cfg in optim_cfgs:
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
        # if len(avg_train_loss) > 1 and torch.trapezoid(avg_train_loss) < best_loss:
        if len(avg_train_loss) > 1 and torch.mean(avg_train_loss[-10:]) < best_loss:
            best_loss = torch.mean(avg_train_loss[-10:])
            best_traj = avg_train_loss
            best_cfg = optim_cfg

    # Collect results for best configuration.
    df = pd.DataFrame(
        {
            "epoch": [i for i in range(len(best_traj))],
            "average_train_loss": [val.item() for val in best_traj],
        }
    )

    path = get_path([var_to_str(model_cfg), optim_cfgs[0]["optimizer"]], out_path=out_path)

    for seed in seeds:
        results = load_results(model_cfg, best_cfg, seed, out_path=out_path)
        df[f"seed_{seed}_train"] = results["metrics"]["train_loss"]
        df[f"seed_{seed}_val"] = results["metrics"]["val_loss"]
        if "nb_checkpoints" in results.keys():
            nb_checkpoints = results["nb_checkpoints"]
            pickle.dump(
                nb_checkpoints, open(os.path.join(path, "nb_checkpoints.p"), "wb")
            )
        if seed == 1 or seed == 0:
            weights = results["weights"]

    print("Saving results to location:")
    print(path)

    pickle.dump(best_cfg, open(os.path.join(path, "best_cfg.p"), "wb"))
    pickle.dump(weights, open(os.path.join(path, "best_weights.p"), "wb"))
    pickle.dump(df, open(os.path.join(path, "best_traj.p"), "wb"))
