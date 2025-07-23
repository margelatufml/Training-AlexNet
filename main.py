import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the training data
train_df = pd.read_csv("train.csv")
# Map author labels to numeric IDs
label_map = {'EAP': 0, 'HPL': 1, 'MWS': 2}
train_df['label'] = train_df['author'].map(label_map)

# Initialize tokenizer for DeBERTa-v3-large
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Utility: Tokenize a list of texts and optionally attach labels
def tokenize_texts(texts, labels=None):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
    if labels is not None:
        encodings["labels"] = list(labels)
    return encodings


# Define a PyTorch Dataset to work with our encodings
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


# Set up 6-fold stratified cross-validation
NUM_FOLDS = 6
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Prepare test data encoding once (to use for each fold’s prediction)
test_df = pd.read_csv("test.csv")
test_enc = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=MAX_LEN)
test_dataset = TextDataset(test_enc)

all_fold_probs = []  # to collect probability predictions from each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), start=1):
    print(f"Training fold {fold}/{NUM_FOLDS}...")
    # Split into training and validation sets for this fold
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    # Tokenize the text and labels
    train_enc = tokenize_texts(train_data['text'], train_data['label'])
    val_enc = tokenize_texts(val_data['text'], val_data['label'])
    # Create Dataset objects for Trainer
    train_dataset = TextDataset(train_enc)
    val_dataset = TextDataset(val_enc)
    # Load a fresh DeBERTa model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    # Setup training hyperparameters (adjust batch size if you encounter OOM errors)
    training_args = TrainingArguments(
        output_dir=f"./deberta_results_fold{fold}",
        num_train_epochs=4,  # 4 epochs (adjust based on validation performance)
        per_device_train_batch_size=32,  # if OOM, try 16 and set grad_accumulation_steps=2
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # lower LR for large models
        weight_decay=0.01,
        evaluation_strategy="epoch",  # evaluate on the validation set each epoch
        save_strategy="no",  # no intermediate checkpoints (to save space)
        load_best_model_at_end=False,  # not saving checkpoints, so set this False
        metric_for_best_model="eval_loss",
        label_smoothing_factor=0.1,  # helps reduce over-confident predictions
        warmup_ratio=0.1,  # warmup 10% of steps for stability
        fp16=True,  # use mixed precision for speed
        seed=42 + fold,  # different seed each fold for diversity
        dataloader_num_workers=4,
        report_to="none"  # no HuggingFace logging (like Wandb) in this context
    )
    # Initialize Trainer with our model and data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    # Train the model on this fold
    trainer.train()
    # Evaluate on validation (optional: Trainer already evaluates each epoch due to eval_strategy="epoch")
    # val_metrics = trainer.evaluate()  # you can print val_metrics if needed

    # Predict on the test set with the current fold's model
    predictions = trainer.predict(test_dataset)  # returns predictions, label_ids, metrics
    logits = predictions.predictions  # model outputs (logits) for each test example
    # Convert logits to probability distributions with softmax
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    all_fold_probs.append(probs)
    print(f"Fold {fold} done.")

# Ensemble: average the probabilities from all folds
avg_probs = np.mean(np.array(all_fold_probs), axis=0)  # shape: [num_test_samples, 3]

# Determine final predicted author for each test sample
pred_label_indices = avg_probs.argmax(axis=1)
# Map label indices back to author names
inv_label_map = {v: k for k, v in label_map.items()}
pred_authors = [inv_label_map[idx] for idx in pred_label_indices]

# Create submission DataFrame
submission = pd.DataFrame({'id': test_df['id'], 'author': pred_authors})
submission.to_csv("submission.csv", index=False)
print("Ensemble submission saved to submission.csv")
