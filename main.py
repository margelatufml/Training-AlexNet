import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# ---- 1. Environment and Logging ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_DIR = "./huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---- 2. Data ----
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)
print(f"Loaded data: train shape = {train.shape}, test shape = {test.shape}")

# ---- 3. Model & Tokenizer ----
MODEL_NAME = "roberta-large"  # Change to another model if you want
MAX_LEN = 384  # Increase to 512 if memory allows
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(tokenizer, df):
    return tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,  # Dynamic padding
        max_length=MAX_LEN,
        return_tensors=None
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

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# ---- 4. Cross-validation ----
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
        per_device_train_batch_size=32,  # RTX 6000 Ada: try 32/48/64, tune for OOM
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f"./logs_fold{fold + 1}",
        eval_strategy="epoch",
        save_strategy="epoch",  # Must match eval_strategy if using load_best_model_at_end
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        fp16=True,  # Always use on RTX 6000 Ada!
        dataloader_num_workers=4,  # RTX 6000 Ada can go 4-8, but higher is not always faster
        logging_steps=25,  # Print progress often
        seed=42 + fold,
        report_to="none"
    )

    print(f"Starting Trainer for fold {fold+1}... Batch size: {training_args.per_device_train_batch_size}")
    try:
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
    except Exception as e:
        print("Training error:", e)
        raise

    val_logits = trainer.predict(val_dataset).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    oof_preds[val_idx] = val_probs

    if fold == 0:
        test_encodings = preprocess(tokenizer, test)
        test_dataset = SpookyDataset(test_encodings)
        test_logits_all = np.zeros((NUM_FOLDS, len(test), 3))
    test_logits = trainer.predict(test_dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_logits_all[fold] = test_probs

    # Cleanup to avoid disk bloat
    import shutil
    shutil.rmtree(f"./results_fold{fold + 1}", ignore_errors=True)
    shutil.rmtree(f"./logs_fold{fold + 1}", ignore_errors=True)
    torch.cuda.empty_cache()

# ---- 5. Final Outputs ----
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
