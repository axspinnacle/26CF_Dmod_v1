import json
import os


def load_config(config_path="../config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def resolve_path(config, section, file_key):
    base = config["global"]["base_data_dir"]
    rel_path = config[section]["files"][file_key]
    return os.path.join(base, rel_path)
