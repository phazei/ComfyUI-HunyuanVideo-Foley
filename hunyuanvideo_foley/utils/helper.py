import collections.abc
from itertools import repeat
import importlib
import yaml
import time

def default(value, default_val):
    return default_val if value is None else value


def default_dtype(value, default_val):
    if value is not None:
        assert isinstance(value, type(default_val)), f"Expect {type(default_val)}, got {type(value)}."
        return value
    return default_val


def repeat_interleave(lst, num_repeats):
    return [item for item in lst for _ in range(num_repeats)]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n - (n % k) + k


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def merge_yaml_files(file_list):
    merged_config = {}

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                # Remove the first level
                for key, value in config.items():
                    if isinstance(value, dict):
                        merged_config = merge_dicts(merged_config, value)
                    else:
                        merged_config[key] = value

    return merged_config


def merge_dict(file_list):
    merged_config = {}

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                merged_config = merge_dicts(merged_config, config)

    return merged_config


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def readable_time(seconds):
    """ Convert time seconds to a readable format: DD Days, HH Hours, MM Minutes, SS Seconds """
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days} Days, {hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if hours > 0:
        return f"{hours} Hours, {minutes} Minutes, {seconds} Seconds"
    if minutes > 0:
        return f"{minutes} Minutes, {seconds} Seconds"
    return f"{seconds} Seconds"


def get_obj_from_cfg(cfg, reload=False):
    if isinstance(cfg, str):
        return get_obj_from_str(cfg, reload)
    elif isinstance(cfg, (list, tuple,)):
        return tuple([get_obj_from_str(c, reload) for c in cfg])
    else:
        raise NotImplementedError(f"Not implemented for {type(cfg)}.")
