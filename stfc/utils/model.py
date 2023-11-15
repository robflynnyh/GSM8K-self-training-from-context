from transformers import AutoTokenizer, GPTNeoXForCausalLM

def load(
        name:str = "EleutherAI/pythia-410m", 
        cache_dir:str = None, 
        in8bit:bool=False,
        use_cache:bool = False
    ) -> tuple([GPTNeoXForCausalLM, AutoTokenizer]):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = cache_dir)
    model = GPTNeoXForCausalLM.from_pretrained(name, cache_dir = cache_dir, load_in_8bit = in8bit, use_cache = use_cache)
    return model, tokenizer