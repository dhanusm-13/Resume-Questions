# evaluate_t5.py (create new file)
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_questions_final")
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_questions_final").to(device)
val_df = pd.read_csv("val_combined.csv")
for i, row in val_df.head(5).iterrows():
    inputs = tokenizer(row["input_text"], return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(inputs["input_ids"], max_length=150)
    print("Input:", row["input_text"][:100], "...")
    print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Expected:", row["target_text"])
    print("-" * 50)