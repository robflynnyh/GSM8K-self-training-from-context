import os, sys, json
from omegaconf import OmegaConf
paths_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paths.yaml")

def load_config(config: str = paths_path):
    config = OmegaConf.load(config)
    return config


def load_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def save_jsonl(data, file):
    with open(file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

