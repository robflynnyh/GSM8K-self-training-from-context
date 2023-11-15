import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(path=None, jsonl=None, multi_step_prompt=None):
    assert path or jsonl, "Must provide path or jsonl"
    examples = read_jsonl(path) if path else jsonl

    multi_step_prompt = multi_step_prompt + "\n" if multi_step_prompt else ""
    for ex in examples:
        ex.update(question="<|endoftext|>" + ex["question"] + "\n" + multi_step_prompt)
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        tokens = qn_tokens + ans_tokens
        num_ans_tokens = len(ans_tokens)
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, num_answer_tokens=num_ans_tokens)
