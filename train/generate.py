import argparse
import stfc
import torch
from tqdm import tqdm
from typing import List, Dict
from stfc.inference import inference
import os

def main(args):
    config = stfc.utils.load_config()
    model, tokenizer = stfc.utils.model.load(name = config.model.name, cache_dir = config.model.cache_dir, in8bit = False, use_cache=True)
    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'Loaded checkpoint from {args.checkpoint}')

    train_path = config.gsm8k.train if 'train.jsonl' not in os.listdir('./data') else './data/train.jsonl'
    print(f'Loading data from {train_path}')
    train_data = stfc.utils.load_jsonl(train_path)

    outputs = inference(model, tokenizer, train_data, mode=args.mode, verbose=True, batch_size=100)
    jsonlf = [
        {'question': output['generated'],
        'answer': output['ground_truth']}
        for output in outputs
    ]
    stfc.utils.general.save_jsonl(jsonlf, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='multistep', choices=['cot', 'vanilla', 'multistep'])
    parser.add_argument('-s', '--save_path', type=str, default='./data/train.jsonl')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoints/model.pt')
    args = parser.parse_args()
    
    main(args)