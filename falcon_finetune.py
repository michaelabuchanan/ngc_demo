import torch
from transformers import AutoTokenizer, FalconForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import transformers
import pandas as pd
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# https://www.kaggle.com/code/harveenchadha/tokenize-train-data-using-bert-tokenizer
def tokenizing(text, tokenizer, chunk_size, maxlen):
    input_ids = []
    tt_ids = []
    at_ids = []

    for i in range(0, len(text), chunk_size):
        text_chunk = text[i:i+chunk_size]
        encs = tokenizer(
                    text_chunk,
                    max_length = 2048,
                    padding='max_length',
                    truncation=True
                    )

        input_ids.extend(encs['input_ids'])
        tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])

    return {'input_ids': input_ids, 'token_type_ids': tt_ids, 'attention_mask':at_ids}

model = "ybelkada/falcon-7b-sharded-bf16"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

falcon_model = FalconForCausalLM.from_pretrained(
    model,
    quantization_config=bb_config,
    use_cache=False,
    low_cpu_mem_usage=True
)

training_args = TrainingArguments(
    output_dir="./finetuned_falcon",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16 = True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    optim = "paged_adamw_8bit"
)

print("Preparing LoRAs...")
falcon_model.gradient_checkpointing_enable()
falcon_model = prepare_model_for_kbit_training(falcon_model)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)

lora_model = get_peft_model(falcon_model, lora_config)

dataset = load_dataset("BI55/MedText", split="train")

df = pd.DataFrame(dataset)
prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp

tokens = tokenizing(list(df["Info"]), tokenizer, 256, 2048)
tokens_dataset = Dataset.from_dict(tokens)
split_dataset = tokens_dataset.train_test_split(test_size=0.2)
split_dataset

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained("./finetuned_falcon")