import os, sys, json
from omegaconf import OmegaConf
paths_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paths.yaml")

def load_config(config: str = paths_path):
    config = OmegaConf.load(config)
    return config

def load_jsonl(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]