# 这个文件的主要内容就是对模型进行并行化的多卡精调
import fire
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LinearLR
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from config.training import train_config
from utils.config_utils import update_config, generate_peft_config, generate_dataset_config
from finetune import find_all_linear_names
from utils.dataset_utils import get_preprocessed_dataset
from utils.utils import myProgressCallback


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
        )
    )
    prepare_model_for_kbit_training(model)
    # Changing model into bf16
    # model.to(torch.bfloat16)
    peft_config = generate_peft_config(train_config, kwargs)
    peft_config.lora_alpha = 16
    peft_config.r = 64
    peft_config.target_modules = find_all_linear_names(model)
    # model = BetterTransformer.transform(model)
    # prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to("cuda")

    # 训练参数
    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=2e-4,
        num_train_epochs=1,  # 3
        weight_decay=1,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=250,  # 500
        per_device_train_batch_size=4,
        disable_tqdm=True,
        logging_steps=1
    )
    dataset_config = generate_dataset_config(train_config, kwargs)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=get_preprocessed_dataset(tokenizer=tokenizer, split="train", dataset_config=dataset_config),
        eval_dataset=get_preprocessed_dataset(tokenizer=tokenizer, split="test", dataset_config=dataset_config),
        optimizers=(optimizer, scheduler),
    )
    trainer.add_callback(myProgressCallback)

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)