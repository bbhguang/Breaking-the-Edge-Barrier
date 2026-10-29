"""Configuration parsing and validation."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "experiments.json"


def validate(config):
    def numeric(value, name):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    epsilons = [numeric(e, "epsilon") for e in config["epsilons"]]
    if not epsilons or any(e <= 0 for e in epsilons):
        raise ValueError("epsilons must contain finite positive total budgets")
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("epsilons must be distinct")
    config["epsilons"] = epsilons
    for key in ("trials", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("delta_total", "degree_epsilon_fraction", "degree_delta_fraction"):
        config[key] = numeric(config[key], key)
        if not 0 < config[key] < 1:
            raise ValueError(f"{key} must be between zero and one")
    config["degree_quantile"] = numeric(config["degree_quantile"], "degree_quantile")
    if not 0 < config["degree_quantile"] <= 1:
        raise ValueError("degree_quantile must be in (0, 1]")
    if not config["datasets"]:
        raise ValueError("at least one dataset is required")
    for name, path in config["datasets"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("dataset names must be simple portable identifiers")
        if not isinstance(path, str) or not path:
            raise ValueError("dataset paths must be nonempty strings")
    return config


def load_config(path=DEFAULT_CONFIG):
    return validate(json.loads(Path(path).read_text()))
