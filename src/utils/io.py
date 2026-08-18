import yaml
from pathlib import Path


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
