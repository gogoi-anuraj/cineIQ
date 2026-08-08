import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
import time

def load_imdb_reviews(path: str) -> pd.DataFrame:
    imdb = pd.read_parquet(path)
    return imdb

def split_imdb(imdb: pd.DataFrame, test_size=0.15, random_state=42):
    """
    Simple random split — no temporal ordering needed here, this is a
    generic sentiment classifier train set, unrelated to MovieLens'
    temporal ratings split.
    """
    train_df, val_df = train_test_split(
        imdb, test_size=test_size, random_state=random_state, stratify=imdb["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class IMDBDataset(Dataset):
    """
    Tokenizes on the fly (rather than pre-tokenizing the whole dataset
    upfront) — simpler to implement correctly, and 49,582 reviews is
    small enough that on-the-fly tokenization isn't a bottleneck.
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    return model

def train_distilbert(model, train_loader, val_loader, num_epochs=3, lr=2e-5, device="cpu"):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, n_batches = 0.0, 0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        train_loss = total_train_loss / n_batches

        # Validation
        model.eval()
        total_val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"epoch {epoch+1}/{num_epochs} — train loss: {train_loss:.4f} — "
              f"val loss: {val_loss:.4f} — val acc: {val_acc:.4f} — {time.time()-epoch_start:.1f}s")

    return model, history


if __name__ == "__main__":
    imdb = load_imdb_reviews("data/interim/imdb_reviews_clean.parquet")
    imdb["label"] = (imdb["sentiment"] == "positive").astype(int)

    train_df, val_df = split_imdb(imdb)
    print("train:", train_df.shape, "val:", val_df.shape)
    print("train label balance:", train_df["label"].value_counts(normalize=True).to_dict())
    print("val label balance:", val_df["label"].value_counts(normalize=True).to_dict())

    # Load DistilBERT's tokenizer, check what it does to a sample review
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    sample_text = train_df["review_clean"].iloc[0]
    print("\nsample text (first 200 chars):", sample_text[:200])

    tokenized = tokenizer(sample_text, truncation=True, padding="max_length", max_length=256)
    print("\ntokenized keys:", tokenized.keys())
    print("input_ids length:", len(tokenized["input_ids"]))
    print("first 20 input_ids:", tokenized["input_ids"][:20])
    print("decoded back:", tokenizer.decode(tokenized["input_ids"][:20]))

    train_dataset = IMDBDataset(train_df["review_clean"], train_df["label"], tokenizer, max_length=256)
    val_dataset = IMDBDataset(val_df["review_clean"], val_df["label"], tokenizer, max_length=256)

    print("train_dataset size:", len(train_dataset))
    print("val_dataset size:", len(val_dataset))

    sample = train_dataset[0]
    print("\nsample item keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("label:", sample["label"].item())

    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    print("\nbatch input_ids shape:", batch["input_ids"].shape)
    print("batch attention_mask shape:", batch["attention_mask"].shape)
    print("batch labels shape:", batch["labels"].shape if "labels" in batch else batch["label"].shape)

    model = load_model()
    print(model.config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntotal parameters: {n_params:,}")

    # Forward pass sanity check on the real batch from Part 3
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    print("\nlogits shape:", outputs.logits.shape)  # expect [batch_size, 2]
    print("sample logits:", outputs.logits[0])

    predicted_labels = outputs.logits.argmax(dim=1)
    print("predicted labels (untrained, should look arbitrary):", predicted_labels[:5].tolist())
    print("actual labels:", batch["label"][:5].tolist())

    small_train = train_dataset.texts.iloc[:200]
    small_train_labels = train_dataset.labels.iloc[:200]
    small_val = val_dataset.texts.iloc[:50]
    small_val_labels = val_dataset.labels.iloc[:50]

    small_train_ds = IMDBDataset(small_train, small_train_labels, tokenizer, max_length=256)
    small_val_ds = IMDBDataset(small_val, small_val_labels, tokenizer, max_length=256)

    small_train_loader = DataLoader(small_train_ds, batch_size=8, shuffle=True)
    small_val_loader = DataLoader(small_val_ds, batch_size=8)

    model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model, history = train_distilbert(model, small_train_loader, small_val_loader, num_epochs=1, device=device)