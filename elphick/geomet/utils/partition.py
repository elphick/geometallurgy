import importlib
import inspect
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd


PARTITION_MODELS: dict[str, Callable] = {}


def register_partition_model(name: str, function: Callable, replace: bool = False):
    """Register a partition callable under a name usable in flowsheet YAML."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Partition model name must be a non-empty string")
    if not callable(function):
        raise TypeError(f"Partition model '{name}' must be callable")
    if name in PARTITION_MODELS and not replace:
        raise ValueError(f"Partition model '{name}' is already registered")
    PARTITION_MODELS[name] = function


def resolve_partition_function(partition: dict) -> Callable:
    """Resolve and partially bind a validated partition configuration."""
    if not isinstance(partition, dict):
        raise ValueError("Partition configuration must be a mapping")
    model = partition.get('model')
    module = partition.get('module')
    function_name = partition.get('function')
    if model is not None and (module is not None or function_name is not None):
        raise ValueError("Partition configuration cannot specify both 'model' and 'module'/'function'")
    if model is not None:
        if not isinstance(model, str) or model not in PARTITION_MODELS:
            raise ValueError(f"Unknown partition model '{model}'")
        function = PARTITION_MODELS[model]
    elif module is not None or function_name is not None:
        if not isinstance(module, str) or not isinstance(function_name, str):
            raise ValueError("Partition 'module' and 'function' must both be specified as strings")
        function = load_partition_function(module, function_name)
    else:
        raise ValueError("Partition configuration requires 'model' or both 'module' and 'function'")

    if not callable(function):
        raise TypeError(f"Partition function for model '{model or function_name}' is not callable")
    args = partition.get('args') or {}
    if not isinstance(args, dict):
        raise ValueError("Partition 'args' must be a mapping")
    try:
        inspect.signature(function).bind_partial(**args)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid arguments for partition model '{model or function_name}': {exc}") from exc
    return partial(function, **args) if args else function


def perfect(x: np.ndarray, d50: float) -> np.ndarray:
    """A perfect partition
    
    Args:
        x: The input dimension, e.g. size or density
        d50: The cut-point

    Returns:

    """
    pn: np.ndarray = np.where(x >= d50, 1.0, 0.0)
    return pn


def napier_munn(x: np.ndarray, d50: float, ep: float) -> np.ndarray:
    """The Napier-Munn partition (1998)

    REF: https://www.sciencedirect.com/science/article/pii/S1474667016453036

    Args:
        x: The input dimension, e.g. size or density
        d50: The cut-point
        ep: The Escarte Probable

    Returns:

    """
    pn: np.ndarray = 1 / (1 + np.exp(1.099 * (d50 - x) / ep))
    return pn


def napier_munn_size(size: np.ndarray, d50: float, ep: float) -> np.ndarray:
    return napier_munn(size, d50, ep)


def napier_munn_density(density: np.ndarray, d50: float, ep: float) -> np.ndarray:
    return napier_munn(density, d50, ep)


napier_munn_size_1mm = partial(napier_munn_size, d50=1.0, ep=0.1)


def load_partition_function(module_name, function_name):
    module = importlib.import_module(module_name)
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(f"Partition function '{function_name}' not found in module '{module_name}'") from exc
    if not callable(function):
        raise TypeError(f"Partition function '{module_name}.{function_name}' is not callable")
    return function


for _name, _function in {
    'perfect': perfect,
    'napier_munn': napier_munn,
    'napier_munn_size': napier_munn_size,
    'napier_munn_density': napier_munn_density,
    'napier_munn_size_1mm': napier_munn_size_1mm,
}.items():
    register_partition_model(_name, _function)

# if __name__ == '__main__':
#     da = np.arange(0, 10)
#     PN = perfect(da, d50=6.3)
#     df = pd.DataFrame([da, PN], index=['da', 'pn']).T
#     print(df)
#
#     da = np.arange(0, 10)
#     PN = napier_munn(da, d50=6.3, ep=0.1)
#     df = pd.DataFrame([da, PN], index=['da', 'pn']).T
#     print(df)
