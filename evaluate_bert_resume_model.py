import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import ast
import numpy as np
from tqdm import tqdm
import joblib

# --- Load the saved model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "resume_bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# --- Load and preprocess the dataset ---
df = pd.read_csv("labeled_resumes_for_bert.csv")

# Convert stringified list to actual list
df['labels'] = df['labels'].apply(ast.literal_eval)

mlb = joblib.load("label_binarizer.pkl")
binary_labels = mlb.transform(df["labels"])

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

eval_dataset = ResumeDataset(df["Resume"], binary_labels)
eval_loader = DataLoader(eval_dataset, batch_size=4)

# --- Run Evaluation ---
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = (torch.sigmoid(logits) > 0.5).float()

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(inputs['labels'].cpu().numpy())

# --- Metrics ---
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

print("\n--- Evaluation Results ---")
print(f"Accuracy       : {acc:.4f}")
print(f"F1 Score (micro): {f1:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
