"""
Fabrication process model definitions and configurations.

This module automatically discovers and loads all model definitions from
Python files in the models/ directory.
"""

import importlib
from pathlib import Path

from .base import Fab, Model

models = {}


def _load_models_from_module(module):
    """
    Load all Model instances from a module into the models dict.

    If the module defines a __models__ dict, use that for registration.
    Otherwise, auto-register all Model instances using their variable names.

    Parameters
    ----------
    module : module
        Python module containing Model instances to register.
    """
    if hasattr(module, "__models__"):
        models_dict = getattr(module, "__models__")
        for name, obj in models_dict.items():
            if isinstance(obj, Model):
                models[name] = obj
    else:
        for name, obj in vars(module).items():
            if isinstance(obj, Model):
                models[name] = obj


models_dir = Path(__file__).parent

for model_file in models_dir.glob("*.py"):
    if model_file.stem in ("__init__", "base"):
        continue

    module_name = f".{model_file.stem}"
    try:
        module = importlib.import_module(module_name, package=__package__)
        _load_models_from_module(module)
    except Exception:
        pass


__all__ = [
    "Fab",
    "Model",
    "models",
]
