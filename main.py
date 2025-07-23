import os
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          DataCollatorWithPadding, EarlyStoppingCallback)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch

# HuggingFace cache
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_cache'
os.makedirs('/workspace/huggingface_cache', exist_ok=True)

# 1. Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

MODEL_NAME = "albert-xxlarge-v2"
MAX_LEN = 512  # reduce to 384 if OOM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(tokenizer, df):
    return tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,       # dynamic padding
        max_length=MAX_LEN,
        return_tensors=None  # tensors in collator
    )

class SpookyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    return {"log_loss": log_loss(labels, probs)}

# 2. Data Collator (fast dynamic padding)
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# 3. KFold
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
    print(f"\n==== Fold {fold+1}/{NUM_FOLDS} ====")
    train_fold = train.iloc[train_idx].reset_index(drop=True)
    val_fold = train.iloc[val_idx].reset_index(drop=True)

    train_encodings = preprocess(tokenizer, train_fold)
    val_encodings = preprocess(tokenizer, val_fold)

    train_dataset = SpookyDataset(train_encodings, train_fold['label'].values)
    val_dataset = SpookyDataset(val_encodings, val_fold['label'].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold + 1}",
        num_train_epochs=6,
        per_device_train_batch_size=24,  # ALBERT-xxlarge can be memory-hungry; adjust as needed
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # 1e-5 or 2e-5; ALBERT likes slightly lower LR
        weight_decay=0.01,
        logging_dir=f"./logs_fold{fold + 1}",
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        fp16=True,
        dataloader_num_workers=8,
        seed=42 + fold,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )

    trainer.train()

    val_logits = trainer.predict(val_dataset).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    oof_preds[val_idx] = val_probs

    # test only tokenize ONCE, not per fold!
    if fold == 0:
        test_encodings = preprocess(tokenizer, test)
        test_dataset = SpookyDataset(test_encodings)
        test_logits_all = np.zeros((NUM_FOLDS, len(test), 3))
    test_logits = trainer.predict(test_dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_logits_all[fold] = test_probs

# Average test predictions over folds
test_preds = np.mean(test_logits_all, axis=0)
eps = 1e-15
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
test_preds = np.clip(test_preds, eps, 1 - eps)
oof_logloss = log_loss(train['label'].values, oof_preds)
print(f"\n==== OOF LOGLOSS (CV estimate): {oof_logloss:.5f} ====")

# Submission
sub = pd.DataFrame(test_preds, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test['id'])
sub.to_csv("submission.csv", index=False, float_format="%.12f")
print("submission.csv written.")

oof_df = pd.DataFrame(oof_preds, columns=[label2author[i] for i in range(3)])
oof_df["id"] = train["id"]
oof_df["true_label"] = train["author"]
oof_df.to_csv("oof_predictions.csv", index=False)
print("oof_predictions.csv written.")
