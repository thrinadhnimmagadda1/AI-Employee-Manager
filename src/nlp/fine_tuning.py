"""
BERT Fine-Tuning Training Loop
Uses HuggingFace Trainer API. Called by emotion_classifier.fine_tune_emotion_classifier().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "bert-base-uncased"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    accuracy = float(np.mean(predictions == labels))

    logger.info(f"Eval — accuracy: {accuracy:.4f} | F1 macro: {f1_macro:.4f} | F1 weighted: {f1_weighted:.4f}")
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ─────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────

def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, label_col: str) -> Dataset:
    def tokenize_fn(examples: dict) -> dict:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False,  # DataCollator handles dynamic padding
        )
        tokenized["labels"] = examples[label_col]
        return tokenized

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def run_fine_tuning(
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_labels: int,
    label_col: str = "mapped_label",
    output_dir: Optional[Path] = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
) -> dict:
    """
    Fine-tune BERT on a classification task.

    Args:
        train_dataset:  HuggingFace Dataset with 'text' and label_col columns.
        val_dataset:    Validation Dataset.
        num_labels:     Number of output classes.
        label_col:      Name of the integer label column.
        output_dir:     Where to save the fine-tuned model.
        num_epochs:     Training epochs.
        learning_rate:  AdamW learning rate.
        batch_size:     Per-device batch size.
    """
    output_dir = output_dir or MODELS_DIR / "emotion_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    logger.info("Tokenizing datasets …")
    train_tokenized = _tokenize_dataset(train_dataset, tokenizer, label_col)
    val_tokenized = _tokenize_dataset(val_dataset, tokenizer, label_col)

    logger.info(f"Loading model: {BASE_MODEL} with {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",  # disable wandb/tensorboard unless configured
        fp16=False,  # set True if GPU with half-precision support
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info(f"Starting fine-tuning for {num_epochs} epochs …")
    trainer.train()

    # Final evaluation and classification report
    logger.info("Running final evaluation …")
    eval_results = trainer.evaluate()
    logger.info(f"Final eval results: {eval_results}")

    # Save model and tokenizer
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Generate classification report on validation set
    predictions_output = trainer.predict(val_tokenized)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    report = classification_report(labels, preds, zero_division=0)
    logger.info(f"Classification Report:\n{report}")

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)
    logger.info(f"Classification report saved to {report_path}")

    return {
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_weighted": eval_results.get("eval_f1_weighted", 0.0),
        "accuracy": eval_results.get("eval_accuracy", 0.0),
    }
