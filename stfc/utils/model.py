from transformers import GPTNeoXForCausalLM, AutoTokenizer

def load(name = "EleutherAI/pythia-1b", cache_dir = None):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = cache_dir)
    model = GPTNeoXForCausalLM.from_pretrained(name, cache_dir = cache_dir)
    return tokenizer, model