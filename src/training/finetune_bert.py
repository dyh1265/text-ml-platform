"""Fine-tune BERT on IMDb sentiment data.

Reads silver JSONL from MinIO, fine-tunes BertForSequenceClassification,
and saves the model for use in embedding and inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from src.utils.s3_client import get_object_body, list_objects
from src.utils.structured_logging import log_event

DEFAULT_BASE_MODEL = "textattack/bert-base-uncased-SST-2"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_MAX_LENGTH = 512


def load_silver_records(silver_prefix: str) -> list[dict]:
    """Load silver JSONL records from MinIO under the given prefix."""
    import json

    objects = list_objects(silver_prefix)
    records = []
    for obj in objects:
        key = obj["Key"]
        if not key.endswith(".jsonl"):
            continue
        body = get_object_body(key)
        for line in body.decode("utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class SilverDataset(Dataset):
    """PyTorch Dataset for silver (text, label) pairs."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = [str(r.get("text", "")) for r in records]
        self.labels = [int(r.get("label", 0)) for r in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_finetune(
    *,
    silver_train_prefix: str,
    silver_test_prefix: str | None,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    device: str,
) -> None:
    """Fine-tune BERT on silver train data, optionally evaluate on test."""
    device_obj = torch.device(device)
    log_event("finetune_started", base_model=base_model, epochs=epochs, batch_size=batch_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = BertForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model.to(device_obj)

    train_records = load_silver_records(silver_train_prefix)
    if not train_records:
        raise ValueError(f"No silver records found at {silver_train_prefix}. Run silver job first.")
    log_event("finetune_train_loaded", records=len(train_records), prefix=silver_train_prefix)

    train_ds = SilverDataset(train_records, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            labels = batch["labels"].to(device_obj)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = out.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        log_event("finetune_epoch", epoch=epoch + 1, loss=avg_loss, train_accuracy=train_acc)

    if silver_test_prefix:
        test_records = load_silver_records(silver_test_prefix)
        if test_records:
            test_ds = SilverDataset(test_records, tokenizer, max_length=max_length)
            test_loader = DataLoader(test_ds, batch_size=batch_size)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device_obj)
                    attention_mask = batch["attention_mask"].to(device_obj)
                    labels = batch["labels"].to(device_obj)
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = out.logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            test_acc = correct / total if total else 0.0
            log_event("finetune_test_accuracy", test_accuracy=test_acc, records=total)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_event("finetune_complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDb silver data.")
    parser.add_argument("--silver-train-prefix", type=str, default="silver/imdb/train/")
    parser.add_argument("--silver-test-prefix", type=str, default="silver/imdb/test/")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default="models/bert_sentiment_imdb")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_finetune(
        silver_train_prefix=args.silver_train_prefix,
        silver_test_prefix=args.silver_test_prefix,
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
