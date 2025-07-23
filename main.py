import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)

# Diagnostics: print device info up front
cuda = torch.cuda.is_available()
print(f"\nCUDA available: {cuda}")
if cuda:
    print("Num GPUs:", torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
else:
    print("WARNING: No GPU detected! Training will be much slower.")

# Data
train_df = pd.read_csv("train.csv")
label_map = {'EAP': 0, 'HPL': 1, 'MWS': 2}
train_df['label'] = train_df['author'].map(label_map)
test_df = pd.read_csv("test.csv")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 512
BATCH_SIZE = 128    # Go big! Try 256/192/128, tune as needed for your VRAM
NUM_WORKERS = 16    # Use a high number for best data pipeline
NUM_FOLDS = 6
EPOCHS = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_texts(texts, labels=None):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',  # maximize GPU throughput!
        max_length=MAX_LEN
    )
    if labels is not None:
        encodings["labels"] = list(labels)
    return encodings

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

# Encode test set once
test_enc = tokenizer(
    list(test_df['text']),
    truncation=True,
    padding='max_length',
    max_length=MAX_LEN
)
test_dataset = TextDataset(test_enc)

all_fold_probs = []
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), start=1):
    print(f"\n==== Training fold {fold}/{NUM_FOLDS} ====")
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    train_enc = tokenize_texts(train_data['text'], train_data['label'])
    val_enc = tokenize_texts(val_data['text'], val_data['label'])
    train_dataset = TextDataset(train_enc)
    val_dataset = TextDataset(val_enc)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # bfloat16 is even faster/better than fp16 on Ada, Hopper, A100, H100, L40S, RTX 6000 Ada, etc.
    bf16_flag = torch.cuda.is_bf16_supported() if cuda else False
    print(f"  > bf16 available: {bf16_flag}")

    training_args = TrainingArguments(
        output_dir=f"./deberta_results_fold{fold}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        label_smoothing_factor=0.1,
        warmup_ratio=0.1,
        fp16=not bf16_flag,    # Use fp16 only if bf16 is not supported
        bf16=bf16_flag,        # Use bf16 if available (for Ada/A100 etc.)
        dataloader_num_workers=NUM_WORKERS,
        seed=42 + fold,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    all_fold_probs.append(probs)
    print(f"Fold {fold} done. (Probabilities shape: {probs.shape})")

# Average (ensemble) probabilities
avg_probs = np.mean(np.array(all_fold_probs), axis=0)
pred_label_indices = avg_probs.argmax(axis=1)
inv_label_map = {v: k for k, v in label_map.items()}
pred_authors = [inv_label_map[idx] for idx in pred_label_indices]
submission = pd.DataFrame({'id': test_df['id'], 'author': pred_authors})
submission.to_csv("submission.csv", index=False)
print("\n🔥🔥🔥 Ensemble submission saved to submission.csv 🔥🔥🔥")
