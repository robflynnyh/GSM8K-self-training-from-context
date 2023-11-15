import argparse
import stfc
import torch
from torch.optim import Adam
from tqdm import tqdm
from typing import List, Dict

from stfc.utils import GSMDataset, get_examples

from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main():
    config = stfc.utils.load_config()

    train_path = config.gsm8k.train if 'train.jsonl' not in os.listdir('./data') else './data/train.jsonl'
    print(f'Loading data from {train_path}')
    train_examples = get_examples(train_path, multi_step_prompt="[[Let's try again]]")
    model, tokenizer = stfc.utils.model.load(name = config.model.name, cache_dir = config.model.cache_dir, in8bit = False)
    if "checkpoint" in config.model and os.path.exists(config.model.checkpoint):
        model.load_state_dict(torch.load(config.model.checkpoint))
        print(f'Loaded checkpoint from {config.model.checkpoint}')
    
    train_dset = GSMDataset(tokenizer, train_examples)
    
    device = torch.device("cuda")
    model.to(device)
    model.train()


    model.gradient_checkpointing_enable()

    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    optim = Adam(model.parameters(), lr=1e-7)
    num_epochs = 5

    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)
    num_training_steps = num_epochs * len(train_loader)

 
    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            num_ans_tokens = batch["num_answer_tokens"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = torch.nn.functional.cross_entropy(
                input = logits[:, -num_ans_tokens-1:-1].transpose(1, 2),
                target = input_ids[:, -num_ans_tokens:],
            )
            accelerator.backward(loss)
            optim.step()
            
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss:.5f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    #print(unwrapped_model.generate)

    accelerator.save(unwrapped_model.state_dict(),
        './checkpoints/model.pt'
    )



if __name__ == "__main__":
    main()
