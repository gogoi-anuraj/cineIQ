"""
gru_model.py

Sequential signal — GRU4Rec-style model, final architecture:
  - multi-position training (every prefix -> next-item pair, not just
    the sequence's final item)
  - dropout on embeddings + pre-output, weight decay, early stopping
    (fixes the overfitting found in the naive v1 version)
  - weight tying between the embedding layer and the output layer
    (~65% fewer parameters than an untied output layer, same or better
    Hit Rate@10/NDCG@10)
  - seen-item masking at inference time (a trained model will otherwise
    re-recommend movies already in the user's history — confirmed via
    manual inspection of example outputs, fixed here as a hard
    requirement of prediction, not an optional flag)

Output is a predicted-next-movie score per user, used as one of the
meta-model's three input features. See
score_candidates_for_user() for the per-(user, candidate) scoring
interface the meta-model actually calls.
"""

import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Global seed. Without this, model init/dropout/batch-shuffling are
    unseeded and re-running training produces genuinely different results
    (confirmed empirically: Hit Rate@10 varied 3.18%-3.52% across runs of
    the identical v5 config before this was added)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Movie ID encoding + sequence building
# ---------------------------------------------------------------------------

def build_movie_id_mapping(ratings_train: pd.DataFrame, item_col="movieId"):
    """Maps each movieId to a dense integer index, reserving index 0 for
    padding. Must be saved and reused identically at inference time."""
    unique_movies = sorted(ratings_train[item_col].unique())
    movie_to_idx = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
    return movie_to_idx, idx_to_movie


def build_sequences_encoded(ratings_train: pd.DataFrame, movie_to_idx: dict,
                             user_col="userId", item_col="movieId", timestamp_col="timestamp",
                             max_seq_len: int = 100) -> dict:
    """Chronological per-user sequences of encoded movie indices, capped
    at max_seq_len (keeps the most RECENT items for users with longer
    histories)."""
    sorted_ratings = ratings_train.sort_values([user_col, timestamp_col])
    sequences = {}
    for user_id, group in sorted_ratings.groupby(user_col):
        encoded = [movie_to_idx[m] for m in group[item_col].tolist()]
        if len(encoded) > max_seq_len:
            encoded = encoded[-max_seq_len:]
        sequences[user_id] = encoded
    return sequences


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SequenceDatasetMultiPos(Dataset):
    """
    Multiple training examples per user: every prefix -> next-item pair,
    up to max_examples_per_user positions sampled uniformly at random.
    This is the training data format used for the final (v5) model —
    empirically generalized better than using only each user's single
    final item (v1), once combined with regularization.
    """
    def __init__(self, sequences: dict, min_len: int = 2, max_examples_per_user: int = 20,
                 random_state: int = 42):
        rng = np.random.RandomState(random_state)
        self.examples = []
        for user_id, seq in sequences.items():
            if len(seq) < min_len:
                continue
            valid_positions = list(range(1, len(seq)))
            if max_examples_per_user is not None and len(valid_positions) > max_examples_per_user:
                valid_positions = sorted(
                    rng.choice(valid_positions, size=max_examples_per_user, replace=False)
                )
            for i in valid_positions:
                self.examples.append((seq[:i], seq[i]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_seq, target = self.examples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    input_seqs, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in input_seqs])
    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded_inputs, lengths, targets


# ---------------------------------------------------------------------------
# Model — final architecture (weight-tied, regularized)
# ---------------------------------------------------------------------------

class GRU4RecTied(nn.Module):
    """
    Final GRU4Rec architecture (v5): embedding -> dropout -> GRU ->
    dropout -> projection -> tied-weight output (dot product against the
    embedding matrix + a learned per-movie bias), instead of a separate
    Linear(hidden_dim, vocab_size) layer. Cuts parameter count by ~65%
    vs. an untied output layer, with equal-or-better Hit Rate@10/NDCG@10
    in testing.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Identity() if hidden_dim == embedding_dim
            else nn.Linear(hidden_dim, embedding_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def embed_sequence(self, input_seqs, lengths):
        embedded = self.embedding_dropout(self.embedding(input_seqs))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.projection(self.output_dropout(hidden[-1]))  # [batch, embedding_dim]

    def score_full_vocab(self, projected):
        return projected @ self.embedding.weight.T + self.output_bias  # [batch, vocab_size]

    def forward(self, input_seqs, lengths):
        return self.score_full_vocab(self.embed_sequence(input_seqs, lengths))


# ---------------------------------------------------------------------------
# Validation loss (held-out generalization check — separate from training data)
# ---------------------------------------------------------------------------

def build_val_loss_examples(sequences_train: dict, ratings_val: pd.DataFrame, movie_to_idx: dict,
                             user_col="userId", item_col="movieId", timestamp_col="timestamp"):
    """(full train sequence -> true first val movie) pairs, encoded.
    Used ONLY to measure generalization during training, never for
    training itself. This check is what originally revealed v1's
    overfitting (val loss near the random baseline) despite a
    reasonable-looking Hit Rate@10 — worth always keeping in the loop."""
    first_val = ratings_val.sort_values(timestamp_col).groupby(user_col).head(1)
    examples = []
    for _, row in first_val.iterrows():
        user_id, true_movie = row[user_col], row[item_col]
        if user_id in sequences_train and true_movie in movie_to_idx:
            examples.append((sequences_train[user_id], movie_to_idx[true_movie]))
    return examples


def compute_val_loss(model, val_examples, device="cpu", batch_size=256):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_examples), batch_size):
            batch = val_examples[i:i + batch_size]
            seqs = [torch.tensor(s, dtype=torch.long) for s, _ in batch]
            targets = torch.tensor([t for _, t in batch], dtype=torch.long).to(device)
            lengths = torch.tensor([len(s) for s in seqs])
            padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
            logits = model(padded, lengths)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(batch)
            n += len(batch)
    model.train()
    return total_loss / n


# ---------------------------------------------------------------------------
# Training (regularized, early-stopped, with per-epoch loss history)
# ---------------------------------------------------------------------------

def train_gru(model, train_loader, val_examples, num_epochs: int = 30, lr: float = 0.001,
              weight_decay: float = 1e-5, patience: int = 3, device: str = "cpu",
              verbose: bool = True):
    """
    Trains with weight decay and early stopping on validation loss —
    restores the best-val-loss checkpoint, not necessarily the final
    epoch's weights. Returns (model, history) where history contains
    per-epoch train_loss/val_loss lists, for plotting.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        epoch_start = time.time()

        for batch_inputs, batch_lengths, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_inputs, batch_lengths)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        val_loss = compute_val_loss(model, val_examples, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"epoch {epoch+1}/{num_epochs} — train loss: {train_loss:.4f} — "
                  f"val loss: {val_loss:.4f} — {time.time()-epoch_start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"restored best model — val loss: {best_val_loss:.4f}")

    return model, history


# ---------------------------------------------------------------------------
# Inference — WITH seen-item masking (required, not optional)
# ---------------------------------------------------------------------------

def predict_top_k(model, sequence: list, k: int = 10, device: str = "cpu",
                   exclude_seen: bool = True) -> list:
    """
    Returns the top-k predicted next movie indices for a user's sequence.

    exclude_seen=True (default, required for any real serving use):
    masks out movies already in the user's history before ranking.
    Confirmed via manual inspection that without this, a substantial
    share of top-10 slots were occupied by already-watched titles —
    masking improved Hit Rate@10 by ~39% relative (3.71% -> 5.14% in
    Week 2 testing) simply by freeing those slots for genuine candidates.
    Only set False for diagnostic/debugging purposes.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        length_tensor = torch.tensor([len(sequence)])
        logits = model(input_tensor, length_tensor).clone()

        if exclude_seen:
            seen = torch.tensor(list(set(sequence)), dtype=torch.long, device=device)
            logits[0, seen] = -float("inf")

        top_k_indices = torch.topk(logits, k=k, dim=1).indices[0].tolist()
    return top_k_indices


def evaluate_gru(model, sequences: dict, ratings_val: pd.DataFrame, movie_to_idx: dict,
                  idx_to_movie: dict, k: int = 10, device: str = "cpu", exclude_seen: bool = True,
                  user_col="userId", item_col="movieId", timestamp_col="timestamp"):
    """Hit Rate@k and NDCG@k on the given val set (pass the WARM-user
    subset for a fair comparison against SVD/Markov — see split.py's
    get_warm_users/filter_to_warm_users)."""
    first_val = (
        ratings_val.sort_values(timestamp_col).groupby(user_col).head(1)
        [[user_col, item_col]].rename(columns={item_col: "true_next_movie"})
    )
    hits, ndcgs, n_evaluated = 0, [], 0
    for _, row in first_val.iterrows():
        user_id, true_movie = row[user_col], row["true_next_movie"]
        if user_id not in sequences or true_movie not in movie_to_idx:
            continue
        top_k_idxs = predict_top_k(model, sequences[user_id], k=k, device=device, exclude_seen=exclude_seen)
        top_k_movies = [idx_to_movie[i] for i in top_k_idxs]
        n_evaluated += 1
        if true_movie in top_k_movies:
            hits += 1
            ndcgs.append(1.0 / np.log2(top_k_movies.index(true_movie) + 2))
        else:
            ndcgs.append(0.0)
    hit_rate = hits / n_evaluated if n_evaluated else 0.0
    ndcg = np.mean(ndcgs) if ndcgs else 0.0
    return hit_rate, ndcg, n_evaluated


# ---------------------------------------------------------------------------
# Meta-model interface — per-(user, candidate) scoring
# ---------------------------------------------------------------------------

def score_candidates_for_user(model, user_sequence: list, candidate_movie_ids: list,
                               movie_to_idx: dict, device: str = "cpu",
                               exclude_seen: bool = True) -> pd.DataFrame:
    """
    Scores a specific list of candidate movies for one user — this is
    what the meta-model (Week 3) calls to get the GRU feature column,
    analogous to svd_model.score_batch() and
    content_model.score_candidates_for_user(). Candidates not in the
    training vocabulary get a score of -inf (never recommend something
    the model has no representation for) unless exclude_seen filtering
    would have removed them anyway.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([user_sequence], dtype=torch.long).to(device)
        length_tensor = torch.tensor([len(user_sequence)])
        logits = model(input_tensor, length_tensor).clone()[0]

        if exclude_seen:
            seen = torch.tensor(list(set(user_sequence)), dtype=torch.long, device=device)
            logits[seen] = -float("inf")

    scores = []
    for movie_id in candidate_movie_ids:
        if movie_id in movie_to_idx:
            idx = movie_to_idx[movie_id]
            scores.append(logits[idx].item())
        else:
            scores.append(float("-inf"))  # unseen in training vocab -> no signal

    return pd.DataFrame({"movieId": candidate_movie_ids, "gru_score": scores})


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_gru_artifacts(model, movie_to_idx, idx_to_movie, model_path, mapping_path,
                        embedding_dim=64, hidden_dim=128, dropout=0.3):
    torch.save(model.state_dict(), model_path)
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "movie_to_idx": movie_to_idx, "idx_to_movie": idx_to_movie,
            "embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "dropout": dropout,
        }, f)
    print(f"saved model to {model_path}")
    print(f"saved mappings to {mapping_path}")


def load_gru_artifacts(model_path, mapping_path, device="cpu"):
    with open(mapping_path, "rb") as f:
        meta = pickle.load(f)
    movie_to_idx, idx_to_movie = meta["movie_to_idx"], meta["idx_to_movie"]
    vocab_size = len(movie_to_idx) + 1

    model = GRU4RecTied(
        vocab_size=vocab_size,
        embedding_dim=meta.get("embedding_dim", 64),
        hidden_dim=meta.get("hidden_dim", 128),
        dropout=meta.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, movie_to_idx, idx_to_movie


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_train", default="data/interim/ratings_train.parquet")
    parser.add_argument("--val_warm", default="data/interim/ratings_val_warm.parquet")
    parser.add_argument("--warm_users", default="data/interim/warm_users.parquet")
    parser.add_argument("--out_model", default="data/interim/gru_model_final.pt")
    parser.add_argument("--out_mapping", default="data/interim/gru_mappings.pkl")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--max_examples_per_user", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    warm_users = set(pd.read_parquet(args.warm_users)["userId"])
    ratings_train = pd.read_parquet(args.ratings_train)
    ratings_train_warm = ratings_train[ratings_train["userId"].isin(warm_users)]
    ratings_val_warm = pd.read_parquet(args.val_warm)
    print("ratings_train_warm shape:", ratings_train_warm.shape)

    movie_to_idx, idx_to_movie = build_movie_id_mapping(ratings_train_warm)
    vocab_size = len(movie_to_idx) + 1
    print("vocab size:", vocab_size)

    sequences = build_sequences_encoded(ratings_train_warm, movie_to_idx, max_seq_len=args.max_seq_len)
    print("sequences:", len(sequences))

    val_loss_examples = build_val_loss_examples(sequences, ratings_val_warm, movie_to_idx)
    print("val loss eval examples:", len(val_loss_examples))

    dataset = SequenceDatasetMultiPos(sequences, max_examples_per_user=args.max_examples_per_user)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print("training examples:", len(dataset))

    model = GRU4RecTied(vocab_size=vocab_size, embedding_dim=args.embedding_dim,
                         hidden_dim=args.hidden_dim, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print("model parameters:", f"{n_params:,}")

    model, history = train_gru(
        model, train_loader, val_loss_examples,
        num_epochs=args.num_epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, device=device,
    )

    hr10, ndcg10, n_eval = evaluate_gru(
        model, sequences, ratings_val_warm, movie_to_idx, idx_to_movie,
        k=10, device=device, exclude_seen=True,
    )
    print(f"\nfinal (masked) Hit Rate@10: {hr10:.4f}   NDCG@10: {ndcg10:.4f}   ({n_eval} users)")

    save_gru_artifacts(
        model, movie_to_idx, idx_to_movie, args.out_model, args.out_mapping,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
    )
