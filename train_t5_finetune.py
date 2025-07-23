import pandas as pd
from datasets import Dataset
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer

#  Load tokenizer and model
model_name = "models/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model and tokenizer loaded on device: {device}")

#  Load datasets (ensure correct column names)
train_df = pd.read_csv("train_combined.csv")
val_df = pd.read_csv("val_combined.csv")

# Check columns
assert 'input_text' in train_df.columns, "Column 'input_text' not found in train.csv"
assert 'target_text' in train_df.columns, "Column 'target_text' not found in train.csv"
assert 'input_text' in val_df.columns, "Column 'input_text' not found in val.csv"
assert 'target_text' in val_df.columns, "Column 'target_text' not found in val.csv"

#  Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

#  Preprocessing
def preprocess(example):
    input_enc = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)
    target_enc = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_tokenized = train_dataset.map(preprocess, batched=True)
val_tokenized = val_dataset.map(preprocess, batched=True)

#  Training Arguments
training_args = TrainingArguments(
    output_dir="./t5_finetuned_questions",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)

# === Train ===
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# === Save model and tokenizer ===
trainer.save_model("./t5_finetuned_questions")
tokenizer.save_pretrained("./t5_finetuned_questions")
print("💾 Model saved to ./t5_finetuned_questions")