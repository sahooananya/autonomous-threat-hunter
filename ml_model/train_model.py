import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset as TorchDataset
import os
import logging

# ========== Configuration ==========
MODEL_NAME = 'distilbert-base-uncased'
SAVED_MODEL_PATH = 'ml_model/saved_transformer/'   # Use local path in PyCharm
DATA_PATH = r"C:\Users\KIIT\PycharmProjects\autonomous-threat-hunter-ai_fullstack\ml_model\data"       # Place your CSV file here

TRAIN_FILE = 'UNSW_NB15_training-set.csv'
LABEL_COLUMN = 'label'
FEATURE_COLUMNS = ['sload', 'dload', 'spkts', 'dpkts', 'proto', 'service', 'state']
N_SAMPLES = 100000
MAX_LENGTH = 128
EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# ========== Dataset Class ==========
class LogDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# ========== Preprocessing ==========
def preprocess_data(df, feature_columns, label_column):
    logging.info("Preprocessing data...")
    usable_cols = [col for col in feature_columns if col in df.columns]
    for col in usable_cols:
        df[col] = df[col].fillna('unknown' if df[col].dtype == 'object' else 0).astype(str)

    df['text_features'] = df[usable_cols].apply(
        lambda x: ' '.join([f"{col}:{val}" for col, val in zip(usable_cols, x)]), axis=1
    )
    texts = df['text_features'].tolist()

    if df[label_column].dtype == 'object':
        labels = LabelEncoder().fit_transform(df[label_column])
    else:
        labels = df[label_column].astype(int).tolist()

    return texts, labels

# ========== Main Training ==========
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_file_path = os.path.join(DATA_PATH, TRAIN_FILE)

    if not os.path.exists(train_file_path):
        logging.error(f"Training file not found: {train_file_path}")
        return

    df = pd.read_csv(train_file_path, nrows=N_SAMPLES)
    if LABEL_COLUMN not in df.columns:
        logging.error(f"Label column '{LABEL_COLUMN}' not found.")
        return

    texts, labels = preprocess_data(df, FEATURE_COLUMNS, LABEL_COLUMN)

    if len(set(labels)) < 2:
        logging.error("Only one class found. Need at least two for classification.")
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = LogDataset(train_encodings, train_labels)
    val_dataset = LogDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_trainer',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    try:
        trainer.train()
    except RuntimeError as e:
        logging.error(f"Training failed: {e}")
        return

    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    model.save_pretrained(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    logging.info("Training complete and model saved.")

if __name__ == '__main__':
    main()
