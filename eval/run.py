import argparse
import stfc
import torch
from tqdm import tqdm
from typing import List, Dict

@torch.no_grad()
def inference(model, tokenizer, data:List[Dict]):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for dp in tqdm(data):
        prompt = dp['question']
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(
            input_ids.to(device),
            early_stopping = True,
            max_new_tokens = 100,
            use_cache = True,
            no_repeat_ngram_size = 3,
        )
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        print(f'\n Q: {prompt} \n A: {gen_text} \n GT: {dp["answer"]}\n')
        print('-'*100)


def main(args):
    config = stfc.utils.load_config()
    model, tokenizer = stfc.utils.model.load(name = config.model.name, cache_dir = config.model.cache_dir, in8bit = True)
    test_data = stfc.utils.load_jsonl(config.gsm8k.test)
    inference(model, tokenizer, test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)