import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import StoppingCriteria, PhrasalConstraint

@torch.no_grad()
def inference(model, tokenizer, data:List[Dict], mode="vanilla", batch_size:int=25, verbose=True):
    outputs = []
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batches = [[]]
    for dp in data:
        prompt = "<|endoftext|>"
        prompt += dp['question']
        if mode == 'cot':
            prompt += f"\n Let's think step by step."
        elif mode == 'multistep':
            prompt += f"\n[[Let's try again]]\n"
        else:
            prompt += "\n"
 
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if len(batches[-1]) == batch_size:
            batches.append([])
        batches[-1].append({
            'prompt': prompt,
            'input_ids': input_ids,
            'answer': dp['answer']
        })
    for batch in tqdm(batches) if verbose else batches:
        max_len = max([len(dp['input_ids'][0]) for dp in batch])
        # pad each input_ids to max_len
        input_ids = torch.stack([torch.cat([torch.ones(max_len - len(dp['input_ids'][0]), dtype=torch.long), dp['input_ids'][0]]) for dp in batch])
        # generate
        gen_tokens = model.generate(
            input_ids.to(device),
            pad_token_id=1,
            max_new_tokens = 80,
            eos_token_id=0,
            num_beams=1, 
            do_sample=False,
            repetition_penalty=1.05, #1.0 means no penalty
        )
        
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for i, dp in enumerate(batch):
            if verbose:
                print(f'\n Q: {dp["prompt"]} \n A: {gen_text[i]} \n GT: {dp["answer"]}\n')
                print('-'*100)
            outputs.append({
                'prompt': dp['prompt'],
                'generated': gen_text[i],
                'ground_truth': dp['answer']
            })

    return outputs