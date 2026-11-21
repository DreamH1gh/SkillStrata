from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["root"] = config_path.parent
    return config


def dataset_config(config: dict, name: str) -> dict:
    if name not in config["datasets"]:
        raise ValueError(f"unknown dataset: {name}")
    value = dict(config["datasets"][name])
    value["path"] = config["root"] / value["path"]
    return value


def model_config(config: dict, name: str) -> dict:
    if name not in config["models"]:
        raise ValueError(f"unknown model: {name}")
    return dict(config["models"][name])
