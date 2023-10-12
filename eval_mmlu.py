import json

import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig
import logging
from tqdm import tqdm
from optimum.bettertransformer import BetterTransformer
from data.mmlu_dataset import mmlu_dataset

from utils.CustomizedLlama import CustomizedLlamaForCausalLM


model_name = '/data/hanzhi/llama-2-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)
mmlu = mmlu_dataset("./data/five_shot_mmlu_test.json", tokenizer=tokenizer)
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    quantization_config=quantization_config
)
#model = BetterTransformer.transform(model)
model.eval()
#model.to("cuda")
generation_config = GenerationConfig(
    max_length=1024,
    max_new_tokens=1,
    eos_token_id=2,
    bos_token_id=1,
)
count_correct = 0
count_total = 0
logging.basicConfig(filename="logfile", level=logging.INFO)
for i in tqdm(range(len(mmlu))):
    generation_result = model.generate(tokenizer.encode(mmlu[i]['input'], return_tensors="pt").to(model.device), generation_config=generation_config)
    result = tokenizer.decode(generation_result[0])
    if result[-1] == mmlu[i]['answer']:
        count_correct += 1
    record = {
        'question': mmlu[i]['input'],
        'expected': mmlu[i]['answer'],
        'generated': result[-1]
    }
    #logging.info(json.dumps(record))
    count_total += 1
    print(count_correct, count_total, count_correct * 1.0 / count_total)
logging.info(f"{count_correct}, {count_total}, {count_correct * 1.0 / count_total}")