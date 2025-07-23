import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm.auto import tqdm
import os
import ast

# --- Load and Prepare Data ---
df = pd.read_csv("labeled_resumes_for_bert.csv")  # ← make sure this file exists
df["labels"] = df["labels"].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df["labels"])
joblib.dump(mlb, "label_binarizer.pkl")

# Tokenization and Dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

X_train, X_val, y_train, y_val = train_test_split(df["Resume"], labels, test_size=0.2, random_state=42)
train_dataset = ResumeDataset(X_train, y_train)
val_dataset = ResumeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "resume_bert_model"

if os.path.exists(model_dir) and os.listdir(model_dir):
    print(f"✅ Model found in '{model_dir}'. Loading and skipping training.")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(device)
    exit()

print("🚀 Starting training from scratch.")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=labels.shape[1],
    problem_type="multi_label_classification"
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# --- Training ---
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (batch_idx + 1) % 20 == 0:
            print(f"🟡 Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    val_preds, val_true, val_loss = [], [], 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            val_loss += outputs.loss.item()
            logits = outputs.logits
            preds = (logits > 0).float().cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels)

    val_preds = np.array(val_preds)
    val_true = np.array(val_true)
    acc = accuracy_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds, average="micro", zero_division=0)
    prec = precision_score(val_true, val_preds, average="micro", zero_division=0)
    rec = recall_score(val_true, val_preds, average="micro", zero_division=0)

    print(f"📊 Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"📈 Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print("-" * 60)

# --- Save Model and Tokenizer ---
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ Training complete. Model saved to '{model_dir}'")
