import argparse

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from Data import load_persuade

MODEL_NAME = "roberta-base" 
NUM_SCORE_LEVELS = 6 
DEMOGRAPHIC_COLUMNS = ["gender", "ell_status", "race_ethnicity", "economically_disadvantaged", "student_disability_status"]

HELD_OUT_TEST_PROMPTS = [
    "Seeking multiple opinions",
    "Community service",
    "Grades for extracurricular activities",
]


class EssayScoreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return {"qwk": qwk}


def build_datasets(csv_path, tokenizer, val_fraction=0.1, seed=42, sample_size=None):
    df = load_persuade(csv_path)
    df = df.dropna(subset=["holistic_essay_score", "full_text"])
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    df["label"] = df["holistic_essay_score"].astype(int) - 1  # 0-indexed

    test_df = df[df["prompt_name"].isin(HELD_OUT_TEST_PROMPTS)].copy()
    trainval_df = df[~df["prompt_name"].isin(HELD_OUT_TEST_PROMPTS)].copy()

    trainval_df = trainval_df.sample(frac=1, random_state=seed)  # shuffle
    n_val = int(len(trainval_df) * val_fraction)
    val_df = trainval_df.iloc[:n_val]
    train_df = trainval_df.iloc[n_val:]

    train_ds = EssayScoreDataset(train_df["full_text"], train_df["label"], tokenizer)
    val_ds = EssayScoreDataset(val_df["full_text"], val_df["label"], tokenizer)

    return train_ds, val_ds, test_df 


def train(csv_path, output_dir="out/marking_model", epochs=3, batch_size=4, sample_size=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_SCORE_LEVELS)

    train_ds, val_ds, test_df = build_datasets(csv_path, tokenizer, sample_size=sample_size)
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Held-out test size: {len(test_df)}")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        logging_steps=50,
        dataloader_num_workers=0, 
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    print("Validation (in-prompt) metrics:", trainer.evaluate())

    if len(test_df) > 0:
        test_ds = EssayScoreDataset(test_df["full_text"], test_df["holistic_essay_score"] - 1, tokenizer)
        print("Cross-prompt TEST metrics:", trainer.evaluate(test_ds))
        fairness_breakdown(trainer, test_df, tokenizer)


def fairness_breakdown(trainer, test_df, tokenizer):
    preds = trainer.predict(EssayScoreDataset(test_df["full_text"], test_df["holistic_essay_score"] - 1, tokenizer))
    pred_labels = np.argmax(preds.predictions, axis=1)
    test_df = test_df.copy()
    test_df["pred_label"] = pred_labels
    test_df["true_label"] = test_df["holistic_essay_score"] - 1

    print("\n--- Fairness breakdown (QWK by subgroup) ---")
    for col in DEMOGRAPHIC_COLUMNS:
        print(f"\nBy {col}:")
        for group_val, group_df in test_df.groupby(col, dropna=True):
            if len(group_df) < 10:
                continue 
            qwk = cohen_kappa_score(group_df["true_label"], group_df["pred_label"], weights="quadratic")
            print(f"  {group_val!r}: n={len(group_df)}, QWK={qwk:.3f}")


def evaluate(csv_path, checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    _, val_ds, test_df = build_datasets(csv_path, tokenizer)

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    print("Validation metrics:", trainer.evaluate(val_ds))

    preds = trainer.predict(val_ds)
    pred_labels = np.argmax(preds.predictions, axis=1)
    print(classification_report(preds.label_ids, pred_labels))

    if len(test_df) > 0:
        fairness_breakdown(trainer, test_df, tokenizer)


def predict_score(essay_text: str, checkpoint_dir: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    inputs = tokenizer(essay_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_label = torch.argmax(logits, dim=1).item()
    return pred_label + 1  # back to 1-6 scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to persuade_2.0_human_scores_demo_id_github.csv")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", default="out/marking_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=None, help="Subsample this many essays for a quick test run (e.g. 500) instead of the full ~26k")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.csv, output_dir=args.checkpoint, epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample_size)
    else:
        evaluate(args.csv, checkpoint_dir=args.checkpoint)