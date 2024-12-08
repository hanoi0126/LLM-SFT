import json
import os
import re
from datetime import datetime

import bitsandbytes as bnb
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

wandb.init(
    project="llm-2024-competition",
    entity="hiroto-weblab",
    name=datetime.now().strftime("%Y-%m-%d/%H-%M-%S"),
)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[-1] if len(names) > 1 else names[0])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def formatting_prompts_func(examples, tokenizer, prompt, eos_token):
    text = prompt.format(examples["text"], examples["output"]) + eos_token
    return {"formatted_text": text}


def main():
    HF_TOKEN = os.getenv("HF_TOKEN")
    base_model_id = "llm-jp/llm-jp-3-13b"
    new_model_id = "llm-jp-3-13b-finetune"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset(
        "json", data_files="data/ichikara-instruction-003-001-1.json"
    )
    prompt = "### 指示\n" "{}\n" "### 回答\n" "{}"
    EOS_TOKEN = tokenizer.eos_token

    dataset = dataset.map(
        lambda examples: formatting_prompts_func(
            examples, tokenizer, prompt, EOS_TOKEN
        ),
        num_proc=4,
    )

    dataset = dataset["train"].train_test_split(test_size=0.1)

    training_arguments = TrainingArguments(
        output_dir=new_model_id,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_strategy="steps",
        logging_steps=10,
        warmup_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_steps=-1,
        learning_rate=5e-5,
        fp16=False,
        bf16=False,
        seed=3407,
        group_by_length=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="formatted_text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    model.config.use_cache = False
    trainer.train()

    datasets = []
    with open("tasks/elyza-tasks-100-TV_0.jsonl", "r") as f:
        item = ""
        for line in f:
            line = line.strip()
            item += line
            if item.endswith("}"):
                datasets.append(json.loads(item))
                item = ""

    results = []
    for data in tqdm(datasets):
        input_text = data["input"]
        test_prompt = "### 指示\n" f"{input_text}" "### 回答"
        tokenized_input = tokenizer.encode(
            test_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        attention_mask = torch.ones_like(tokenized_input)
        with torch.no_grad():
            outputs = model.generate(
                tokenized_input,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        output = tokenizer.decode(
            outputs[tokenized_input.size(1):], skip_special_tokens=True
        )
        results.append(
            {"task_id": data["task_id"], "input": input_text, "output": output}
        )

    jsonl_id = re.sub(".*/", "", new_model_id)
    output_dir = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    with open(f"./{jsonl_id}-outputs.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    model.push_to_hub(new_model_id, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(new_model_id, token=HF_TOKEN, private=True)


if __name__ == "__main__":
    main()
