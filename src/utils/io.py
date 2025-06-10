import os
import pickle
import inspect
import itertools

def get_path(levels, out_path="results/"):
    path = out_path
    for item in levels:
        path = os.path.join(path, item + "/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    return path


def save_results(result, model_cfg, optim_cfg, seed, out_path="results/"):
    path = get_path([var_to_str(model_cfg), var_to_str(optim_cfg)], out_path=out_path)
    f = os.path.join(path, f"seed_{seed}.p")
    print(f"saving results to: '{f}'")
    with open(f, "wb") as fpath:
        pickle.dump(result, fpath)


def load_results(model_cfg, optim_cfg, seed, out_path="results/"):
    # TODO: Make more eld[[degant.
    # The n_class adjustment might need to be re-evaluated or handled upstream if dataset isn't passed.
    # For now, assuming model_cfg might contain dataset info or this is handled before calling.
    # if "iwildcam" in dataset: # This can no longer be done directly here
    #     model_cfg["n_class"] = 60
    # if "amazon" in dataset: # This can no longer be done directly here
    #     model_cfg["n_class"] = 5
    
    # out_path already contains the dataset-specific part from train.py
    path = get_path(
        [var_to_str(model_cfg), var_to_str(optim_cfg)], out_path=out_path
    )
    #print(path,out_path)
    f = os.path.join(path, f"seed_{seed}.p")
    return pickle.load(open(f, "rb"))


def var_to_str(var):
    translate_table = {ord(c): None for c in ",()[]"}
    translate_table.update({ord(" "): "_"})

    if type(var) == dict:
        sortedkeys = sorted(var.keys(), key=lambda x: x.lower())
        var_str = [
            key + "_" + var_to_str(var[key])
            for key in sortedkeys
            if var[key] is not None
        ]
        var_str = "_".join(var_str)
    elif inspect.isclass(var):
        raise NotImplementedError("Do not give classes as items in cfg inputs")
    elif type(var) in [list, set, frozenset, tuple]:
        value_list_str = [var_to_str(item) for item in var]
        var_str = "_".join(value_list_str)
    elif isinstance(var, float):
        var_str = "{0:1.2e}".format(var)
    elif isinstance(var, int):
        var_str = str(var)
    elif isinstance(var, str):
        var_str = var
    elif var is None:
        var_str = str(var)
    else:
        raise NotImplementedError
    return var_str

def dict_to_list(d):
    for key in d:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
    return [dict(zip(d, x)) for x in itertools.product(*d.values())]