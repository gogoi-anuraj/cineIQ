# CINEIQ

An explainable, multi-signal movie recommendation engine combining classical ML, deep sequence modeling, and transformer-based NLP into one coherently evaluated ensemble system.

Demonstrated: collaborative filtering (SVD), content-based filtering (TF-IDF), deep sequential modeling (GRU4Rec-style, with an honest ablation against simpler baselines), transformer-based NLP (DistilBERT sentiment re-ranking), and applied ML rigor (temporal train/val/test evaluation, a validated ensemble, required cold-start handling, and a signal-appropriate explainability layer).

Full write-up of results, methodology, and findings: **[`report/ablation_report.md`](report/ablation_report.md)**.

---

## What it does

Given a user's watch history, CINEIQ produces a ranked list of movie recommendations, each with a human-readable explanation of why it was recommended.

```
SVD + Content + GRU scores → Meta-model (learned weights) → Ranked Top-N
                                                ↓
                          Sentiment re-ranker adjusts ordering within Top-N
                                                ↓
                                    Final recommendation list
```

Users with little rating history (< 25 ratings) are routed around the meta-model entirely and served a content-based + popularity blend instead — SVD and GRU have no meaningful signal for near-empty histories.

---

## Results at a glance

| Model | Hit Rate@10 (validation) | NDCG@10 |
|---|---|---|
| Popularity baseline | 1.32% | 0.67% |
| Markov-chain baseline | 2.11% | 1.00% |
| **GRU (final, seen-item masked)** | **5.14%** | **2.56%** |

| Component | Metric |
|---|---|
| SVD | RMSE 0.813, Precision@10 0.825 |
| Meta-model (logistic regression) | AUC 0.79 |
| Sentiment classifier (DistilBERT) | 92.47% (IMDB val), 82.4% (RT domain-shift) |

See the [ablation report](reports/ablation_report.md) for the full experiment history, including a real overfitting diagnosis-and-fix story for the GRU, a test-split diagnostic explaining a counterintuitive baseline comparison, and every documented limitation.

---

## Project structure

```
cineiq/
├── data/interim/          # processed datasets, trained model artifacts (gitignored)
├── notebooks/             # Kaggle/dev notebooks (data prep, GRU training, sentiment, meta-model)
├── src/
│   ├── data/               # loading, joining, temporal splitting
│   ├── models/             # SVD, content-based, baselines, GRU, meta-model
│   ├── sentiment/           # DistilBERT training + RT review scoring/re-ranking
│   ├── explain/             # per-signal explainability (SVD, content, meta-model, GRU, sentiment)
│   └── serving/              # cold-start routing, unified recommendation pipeline
├── dashboard/               # Streamlit taste dashboard (4 panels + live explainable recs)
├── tests/                    # unit tests (join, split, cold-start logic)
└── reports/
    └── ablation_report.md    # full results write-up
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Note on `scikit-surprise`**: has a C extension that occasionally needs build tools. If the plain install fails:
```bash
pip install numpy Cython
pip install scikit-surprise --no-build-isolation
```

You'll also need the raw datasets (not included in this repo):
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
- [TMDB metadata (The Movies Dataset)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- [IMDB 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Rotten Tomatoes Movies and Critic Reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

## Running the pipeline

Most components (SVD training, GRU training, DistilBERT fine-tuning, meta-model construction) were developed and run on **Kaggle** (GPU + RAM headroom) — see `notebooks/` for the exact, run cell-by-cell versions. The equivalent logic is also available as standalone scripts under `src/`, runnable locally for CPU-bound components:

```bash
# Data prep (Week 1)
python -m src.data.load ...   # see notebooks/01_eda_sparsity_check.ipynb for the full sequence

# SVD (CPU, needs ~8GB+ RAM for the full warm-user set; use --subsample_n_users for local dev)
python -m src.models.svd_model

# Content-based model
python -m src.models.content_model

# GRU (GPU strongly recommended — see notebooks/02_gru_training.ipynb)
python -m src.models.gru_model

# Sentiment classifier (GPU recommended)
python -m src.sentiment.train_classifier

# Apply sentiment classifier to RT reviews
python -m src.sentiment.rerank

# Meta-model (needs SVD/content/GRU artifacts already trained)
python -m src.models.meta_model
```

## Running the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Requires all model artifacts (`svd_model.pkl`, `tfidf_vectorizer.pkl`/`tfidf_matrix.npz`, `gru_model_final.pt`/`gru_mappings.pkl`, `meta_model.pkl`/`meta_model_scaler.pkl`, `movie_sentiment.parquet`) present in `data/interim/`.

## Running tests

```bash
pytest tests/
```

---

## Known limitations

A full list with root-cause explanations is in the [ablation report](reports/ablation_report.md#8-known-limitations-consolidated). The short version:

- **Sentiment re-ranking coverage is uneven across users** — RT review data skews toward mainstream/contemporary titles, so users with arthouse/foreign/older-title taste see little re-ranking benefit.
- **~40% of the movie catalog is excluded as a content-similarity source** — sparse TMDB metadata for those titles makes similarity scores unreliable; they remain eligible as candidates via other signals.
- **The full pipeline underperforms a raw popularity baseline on strict full-catalog Hit Rate@10** on the test split — root-caused via a targeted diagnostic (see report Section 6.2); the meta-model wasn't trained with popularity-weighted negative sampling, which is identified as future work.
- **MLflow is not wired up** — results are currently tracked as plain JSON files per component rather than logged MLflow runs, a gap against the original project scope.

---

## Tech stack

Python · pandas/NumPy · scikit-learn · `Surprise` (SVD) · PyTorch (GRU) · HuggingFace `transformers` (DistilBERT) · Streamlit + Plotly (dashboard) · Kaggle (GPU training)
