# 这个文件的主要内容就是对模型进行并行化的多卡精调
import json
import os
import random
import time
from typing import List, Tuple, Any, Union, Mapping

import deepspeed
import fire
import torch
import torch.optim as optim
import transformers
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LinearLR
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from config.training import train_config
from utils.config_utils import update_config, generate_peft_config, generate_dataset_config
from finetune import find_all_linear_names
from utils.dataset_utils import get_preprocessed_dataset
from utils.utils import myProgressCallback
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from tqdm import tqdm
from utils.CustomizedLlama import CustomizedLlamaForCausalLM


def prepare_input(data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(device)
    return data


def main(**kwargs):
    torch.manual_seed(9999)
    update_config((train_config), **kwargs)
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    prepare_model_for_kbit_training(model)
    # Changing model into bf16
    # model.to(torch.bfloat16)
    peft_config = generate_peft_config(train_config, kwargs)
    peft_config.lora_alpha = 16
    peft_config.r = 64
    #peft_config.target_modules = find_all_linear_names(model)
    # model = BetterTransformer.transform(model)
    # prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # 训练参数
    training_args = TrainingArguments(
        output_dir="/data/hanzhi/output",
        learning_rate=2e-4,
        num_train_epochs=1,  # 3
        weight_decay=train_config.weight_decay,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=100,  # 500
        per_device_train_batch_size=train_config.batch_size_training,
        disable_tqdm=True,
        logging_steps=100,
        dataloader_drop_last=True,
        save_total_limit=40,
    )
    dataset_config = generate_dataset_config(train_config, kwargs)

    optimizer = FusedAdam(model.parameters(), lr=2e-4, betas=(0.9,0.95))
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=6500, num_warmup_steps=650)
    train_dataset = get_preprocessed_dataset(tokenizer=tokenizer, split="train", dataset_config=dataset_config)
    eval_dataset = get_preprocessed_dataset(tokenizer=tokenizer, split="test", dataset_config=dataset_config)
    deepspeed_config = json.load(open("./config/ds.json", "r"))
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config,
        collate_fn=None
    )
    print(deepspeed.checkpointing.is_configured())
    deepspeed.checkpointing.reset()
    for batch in tqdm(train_loader):
        batch = prepare_input(batch, model_engine.device)
        print(model_engine.device)
        loss = model_engine(**batch).loss
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
if __name__ == "__main__":
    fire.Fire(main)
