from transformers import GPTNeoXForCausalLM, AutoTokenizer

def load(name:str = "EleutherAI/pythia-1b", cache_dir:str = None, in8bit:bool=False) -> tuple([GPTNeoXForCausalLM, AutoTokenizer]):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = cache_dir)
    model = GPTNeoXForCausalLM.from_pretrained(name, cache_dir = cache_dir, load_in_8bit = in8bit)
    return model, tokenizer