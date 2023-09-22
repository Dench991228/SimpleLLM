import fire
import torch
from peft import get_peft_model, prepare_model_for_int8_training
from torch import optim
from torch.optim.lr_scheduler import StepLR
from transformers import LlamaForCausalLM, LlamaTokenizer, default_data_collator
from config.training import train_config
from utils.config_utils import generate_peft_config, generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import train, freeze_transformer_layers, check_frozen_layers_peft_model
import transformers



def main(**kwargs):
    update_config((train_config), **kwargs)
    print(train_config.model_name)
    model = LlamaForCausalLM.from_pretrained(train_config.model_name, load_in_8bit=True, use_cache=False)
    # Changing model into bf16
    #model.to(torch.bfloat16)
    peft_config = generate_peft_config(train_config, kwargs)
    prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to("cuda")
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
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
