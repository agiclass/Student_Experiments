import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq, 
    HfArgumentParser,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from data_preprocess import InputOutputDataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    lora_config = LoraConfig(
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj"],
        r=peft_args.lora_rank, 
        lora_alpha=peft_args.lora_alpha, 
        lora_dropout=peft_args.lora_dropout
    )
    model = get_peft_model(model, lora_config).to("cuda")
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True
    )

    if training_args.do_train:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        train_dataset = InputOutputDataset(train_data, tokenizer, data_args)
    if training_args.do_eval:
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = InputOutputDataset(eval_data, tokenizer, data_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )

    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
