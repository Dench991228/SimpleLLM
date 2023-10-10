import fire
import torch
from peft import get_peft_model, prepare_model_for_int8_training, prepare_model_for_kbit_training
from torch import optim
from torch.optim.lr_scheduler import StepLR
from transformers import LlamaForCausalLM, LlamaTokenizer, default_data_collator
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.training import train_config
from utils.CustomizedLlama import CustomizedLlamaForCausalLM
from utils.config_utils import generate_peft_config, generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import train, freeze_transformer_layers, check_frozen_layers_peft_model
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from optimum.bettertransformer import BetterTransformer

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main(**kwargs):
    torch.manual_seed(9999)
    update_config((train_config), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    if train_config.peft_method == "lora":
        model = AutoModelForCausalLM.from_pretrained(train_config.model_name)
        peft_config = generate_peft_config(train_config, kwargs)
        prepare_model_for_kbit_training(model)
        #model.gradient_checkpointing_enable()
        peft_config.lora_alpha=16
        peft_config.r=64
        peft_config.target_modules = find_all_linear_names(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
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
        prepare_model_for_kbit_training(model)
        # Changing model into bf16
        # model.to(torch.bfloat16)
        peft_config = generate_peft_config(train_config, kwargs)
        peft_config.lora_alpha=16
        peft_config.r=64
        peft_config.target_modules = find_all_linear_names(model)
        #model = BetterTransformer.transform(model)
        # prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )
    # Optimizer: Only tuning LoRA parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    result = train(
        model=model,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        train_config=train_config
    )


if __name__ == "__main__":
    fire.Fire(main)
