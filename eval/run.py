import argparse
import stfc
import torch
from tqdm import tqdm
from typing import List, Dict
from stfc.inference import inference

def main(args):
    config = stfc.utils.load_config()
    model, tokenizer = stfc.utils.model.load(name = config.model.name, cache_dir = config.model.cache_dir, in8bit = False, use_cache=True)
    if "checkpoint" in config.model:
        model.load_state_dict(torch.load(config.model.checkpoint))
        print(f'Loaded checkpoint from {config.model.checkpoint}')
    test_data = stfc.utils.load_jsonl(config.gsm8k.train)

    inference(model, tokenizer, test_data, mode=args.mode, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='vanilla', choices=['cot', 'vanilla', 'multistep'])
    args = parser.parse_args()
    
    main(args)