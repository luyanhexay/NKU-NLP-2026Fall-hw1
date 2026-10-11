"""Task 3 - fine-tune bert-base-uncased for NYT classification.

The assignment fixes the three knobs that matter: bert-base-uncased,
max_length = 64, 3 epochs.  Those are the defaults here.

    uv run python -m scripts.task3_bert                        # the reported run
    uv run python -m scripts.task3_bert --max-len 512 --suffix _len512   # appendix

Note on max_length=64: the median NYT document is ~672 tokens, so at 64 the
model sees roughly the first tenth of each article.  The appendix run at 512
exists to quantify that trade-off, not to replace the required configuration.
"""

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.config import ID2LABEL, LABEL2ID, LABELS, RESULTS_DIR, SEED
from src.data import load_splits
from src.metrics import evaluate

MODEL_NAME = "google-bert/bert-base-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0


class NewsDataset(Dataset):
    """NYT documents tokenised to a fixed maximum length."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding=False
        )
        self.labels = [LABEL2ID[lab] for lab in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    """Return predicted class *names*, so callers never mix ids and labels."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    return np.array([ID2LABEL[int(i)] for i in np.concatenate(predictions)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="accumulate gradients over this many batches; combined with "
             "--batch-size it fixes the effective batch size, which keeps the "
             "longer-sequence control runs comparable to the main run",
    )
    parser.add_argument("--suffix", default="")
    parser.add_argument("--name", default="bert")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed for weight init dropout and batch order")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  max_len={args.max_len}  epochs={args.epochs}")

    train, val, test = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)

    loaders = {
        "train": DataLoader(
            NewsDataset(train["text"], train["label"], tokenizer, args.max_len),
            batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        ),
        "val": DataLoader(
            NewsDataset(val["text"], val["label"], tokenizer, args.max_len),
            batch_size=64, shuffle=False, collate_fn=collator,
        ),
        "test": DataLoader(
            NewsDataset(test["text"], test["label"], tokenizer, args.max_len),
            batch_size=64, shuffle=False, collate_fn=collator,
        ),
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=dict(ID2LABEL), label2id=dict(LABEL2ID),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = -(len(loaders["train"]) // -args.grad_accum)   # ceil
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps
    )

    best_state, best_val_f1, best_epoch = None, -1.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss={running_loss / step:.4f}")

        val_f1 = f1_score(
            val["label"], predict(model, loaders["val"], device),
            average="macro", labels=LABELS, zero_division=0,
        )
        print(f"epoch {epoch}: train loss={running_loss / len(loaders['train']):.4f}  "
              f"val macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"best epoch {best_epoch} (val macro-F1={best_val_f1:.4f})")
    model.load_state_dict(best_state)

    test_predictions = predict(model, loaders["test"], device)

    # Save the per-document predictions.  The error analyses in the report
    # (length stratification, worked examples) need to know which document was
    # misclassified, and a confusion matrix cannot tell you that.
    import pandas as pd
    pd.DataFrame({
        "idx": test["idx"].to_numpy(),
        "label": test["label"].to_numpy(),
        "prediction": test_predictions,
    }).to_csv(RESULTS_DIR / f"{args.name}{args.suffix}_test_predictions.csv", index=False)

    evaluate(
        f"{args.name}{args.suffix}",
        test["label"],
        test_predictions,
        meta={
            "model": MODEL_NAME,
            "max_length": args.max_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "truncation": "longest_first (head of the document is kept)",
            "seed": args.seed,
            "best_epoch": best_epoch,
            "val_macro_f1": float(best_val_f1),
        },
    )


if __name__ == "__main__":
    main()
