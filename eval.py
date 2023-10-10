import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig
import logging
from tqdm import tqdm
from optimum.bettertransformer import BetterTransformer

from utils.CustomizedLlama import CustomizedLlamaForCausalLM


def get_mmlu_instruct(mmlu_obj: dict):
    result = mmlu_obj['question']
    choice = ['a.', 'b.', 'c.', 'd.']
    for idx, item in enumerate(mmlu_obj['choices']):
        result = "\n".join([result, choice[idx]+item])
    return result
mmlu = load_dataset("/data/hanzhi/mmlu", 'all', split="test")
model_name = '/data/hanzhi/llama-2-7b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)

model = CustomizedLlamaForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    ))
model = BetterTransformer.transform(model)
model.eval()
generation_config = GenerationConfig(
    max_length=512,
    max_new_tokens=512,
    eos_token_id=2,
    bos_token_id=1,
)
logging.basicConfig(filename="logfile", level=logging.INFO)
for idx, mmlu_obj in tqdm(enumerate(mmlu)):
    instruct = tokenizer(get_mmlu_instruct(mmlu[idx]), return_tensors="pt")['input_ids'].to(model.device)
    result = tokenizer.decode(model.generate(inputs=instruct, generation_config=generation_config)[0])
    logging.info(str(idx)+result)