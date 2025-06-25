import torch, json
from datasets import load_dataset, Features, Value, Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

checkpoint = "Salesforce/blip2-flan-t5-xl"
processor = Blip2Processor.from_pretrained(checkpoint)
model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, load_in_8bit=True)

# LoRA
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05
)
model = get_peft_model(model, lora_cfg)

def encode(batch):
    inputs = processor(
        images=batch["image"],
        text=[f"Question: {q}\nAnswer:" for q in batch["question"]],
        return_tensors="pt", padding="max_length", truncation=True
    )
    with processor.as_target_processor():
        labels = processor(text=batch["answer"], return_tensors="pt", padding="max_length", truncation=True).input_ids
    batch.update(inputs)
    batch["labels"] = labels
    return batch

ds = load_dataset("json", data_files={"train":"data/train.json","validation":"data/val.json"},
                  features=Features({"image": Image(), "question": Value("string"), "answer": Value("string")}))
ds = ds.map(encode, batched=True, remove_columns=["image","question","answer"])

args = TrainingArguments(
    output_dir="outputs/blip2_baseline",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_steps=1000,
    fp16=True,
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
model.save_pretrained("outputs/blip2_baseline/final")