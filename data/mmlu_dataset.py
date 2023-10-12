import json

import torch
from torch.utils.data import Dataset


class mmlu_dataset(Dataset):
    def __init__(self, path: str, tokenizer):
        self.items = []
        self.jsons = open(path)
        for line in self.jsons.readlines():
            self.items.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_word = 512

    def __len__(self):
        return len(self.items)
    def __getitem__(self, item):
        obj = self.items[item]
        # 输入大语言模型的题目
        #print(obj)
        prompt = self.tokenizer.encode(obj["input"])
        prompt.append(self.tokenizer.eos_token_id)
        prompt = torch.tensor(prompt, dtype=torch.int64)
        padding = self.max_word - prompt.shape[0]
        if padding > 0:
            prompt = torch.cat((prompt, torch.zeros(padding, dtype=torch.int64) - 1))
        else:
            prompt = prompt[:self.max_word]
        # 把Attention mask给设计好
        attention_mask = prompt.ge(0)
        # 再把真正的padding给设置回去
        prompt[~attention_mask] = 0
        result = {
            "input_ids": prompt,
            "input": obj['input'],
            "attention_mask": attention_mask.float(),
            "answer": obj['output']
        }
        return result

