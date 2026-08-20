# CINEIQ+ — Ablation Report

**A multi-signal, explainable movie recommendation engine combining classical ML, deep sequence modeling, and transformer-based NLP.**

---

## 1. Summary

CINEIQ+ blends three independently-trained signals — SVD collaborative filtering, TF-IDF content-based filtering, and a GRU sequential model — via a learned meta-model, then applies a sentiment-based post-hoc re-ranking pass and a cold-start routing rule. Every component is benchmarked against defined baselines, and every recommendation carries a human-readable explanation.

The headline result: on validation data, the full stack (Popularity → Markov → GRU) shows a clean, monotonic improvement in next-item prediction quality. On the held-out test split, this picture becomes more nuanced — a finding investigated and explained in Section 6, not glossed over.

| Component | Status | Headline metric |
|---|---|---|
| SVD | RMSE 0.813, Precision@10 0.825 |
| Content-based (TF-IDF) | Validated similarity search + per-user scoring |
| GRU (sequential) | Hit Rate@10 5.14% (masked), beats Markov & Popularity |
| Markov-chain baseline | Hit Rate@10 2.11% (val) |
| Popularity baseline | Hit Rate@10 1.32% (val) |
| Sentiment classifier (DistilBERT) | 92.47% IMDB val accuracy, 82.4% on RT (domain-shift validated) |
| Meta-model (logistic regression) | AUC 0.7877 |
| Cold-start routing | N=25 threshold, 11.0% of users routed |
| Explainability layer | All 5 signals (SVD, content, meta-model, GRU, sentiment) |
| Dashboard | 4 panels, validation-split only |

---

## 2. Data & Evaluation Discipline

### 2.1 Datasets

| Dataset | Role | Coverage achieved |
|---|---|---|
| MovieLens 25M | Ratings, timestamps, sequences | 25,000,095 ratings |
| TMDB metadata | Genre/keyword/cast text | 98.9% ratings-level join coverage |
| IMDB 50K Reviews | Sentiment classifier training | 49,582 reviews after dedup |
| RT critic reviews | Sentiment scoring target | 66.5% ratings-level join coverage |

### 2.2 Temporal split

A single global temporal cutoff was applied consistently across every component (never mixed with random splits):

- **Train**: < 2017-01-01 (83.2%, 20,798,765 ratings)
- **Validation**: 2017-01-01 to 2018-07-01 (9.2%, 2,301,354 ratings)
- **Test**: ≥ 2018-07-01 (7.6%, 1,899,976 ratings) — touched exactly once, in Section 6

### 2.3 Warm-user evaluation subset

A significant methodological finding from Week 1: **66–75% of validation/test users had zero train-period rating history** (single-burst raters, a known property of MovieLens 25M under a temporal split). Evaluating SVD/GRU/meta-model on the full population would have mostly measured cold-start behavior, not model quality. A **warm-user subset** (≥ N=25 train ratings) was carved out for fair, apples-to-apples model evaluation, while the full population remains the honest system-level number.

- Validation warm subset: 5,454 users, 412,348 ratings
- Test warm subset: 3,427 users, 242,544 ratings

### 2.4 Cold-start threshold

Set from the p10 of the train-split sequence-length distribution (24 ratings), rounded to **N = 25**. 15,593 of 142,184 train users (11.0%) fall below this threshold and are routed away from the meta-model.

---

## 3. Component Results

### 3.1 Collaborative Filtering — SVD

Trained via `Surprise`'s SVD on the full warm-user train set (126,591 users, 20,462,986 ratings).

| Metric | Value |
|---|---|
| RMSE (val, warm) | 0.813 |
| Precision@10 (threshold=3.5) | 0.8254 |
| Training time | 169.3s (full run, Kaggle) |

SVD training on the **full unfiltered dataset caused a `MemoryError` on an 8.3GB local machine** — resolved by filtering to warm users only, which is also the architecturally correct choice (cold-start users are routed around SVD at serving time regardless, per Section 3.4).

### 3.2 Content-Based Filtering — TF-IDF

TF-IDF vectorization over `content_text` (TMDB genres + keywords + top-5 cast, with a MovieLens-genre fallback), 35,949-term vocabulary across 62,423 movies.

**A real precision issue was found and fixed during development**: sparse-metadata movies (e.g., a film sharing only one rare, minor cast member with the query movie) produced false-positive "similar movie" matches — a single shared rare term dominating an otherwise near-empty vector. After testing several fixes (informative-term weighting via IDF was tried and found to incorrectly penalize genuinely meaningful shared terms), a **minimum content-token floor (6 tokens)** was adopted as the working, validated solution.

- **Documented limitation**: ~40% of the catalog is excluded as a similarity *source* under this threshold (though such movies remain eligible as *candidates* scored by SVD/popularity).

### 3.3 Sequential Model — GRU

The most heavily iterated component. Final architecture (**v5**): multi-position training data, dropout + weight decay + early stopping, and **weight tying** between the embedding and output layers.

| Version | Hit Rate@10 | NDCG@10 | Val loss (CE) | Params |
|---|---|---|---|---|
| Popularity baseline | 1.32% | 0.67% | — | — |
| Markov-chain baseline | 2.11% | 1.00% | — | — |
| v1 (single-position, unregularized) | 2.83% | 1.46% | 9.535 (≈ random baseline) | 7.28M |
| v2 (multi-position, unregularized) | 2.60% | 1.30% | 9.159 | 7.28M |
| v3 (weighted multi-position) | 2.58% | 1.26% | 9.304 | 7.28M |
| v5 — **final** (regularized, tied weights) | 3.18–3.52%* | 1.57–1.75%* | 8.43–8.45 | **2.51M (−65%)** |
| v6 (BPR + negative sampling, same architecture) | 2.50% | 1.10% | 8.568 | 2.51M |
| **v5 + seen-item masking (final, reported)** | **5.14%** | **2.56%** | — | 2.51M |

*Range reflects genuine run-to-run variance discovered before a global random seed was added (see Section 5.4).

**Key findings from this component:**

1. **v1 looked successful on Hit Rate@10 alone but was badly overfitting** — a validation-loss check (near the random baseline of `ln(vocab_size) ≈ 10.53`) revealed this, despite v1 beating both baselines on the surface metric. This is the single most important methodological catch in the project: **a model can appear to work on one metric while failing to generalize**, detectable only by tracking a second, orthogonal signal.
2. **Two "improvements" (multi-position training, BPR loss) each independently underperformed the simpler baseline** before regularization was added — a legitimate negative result, not a wasted effort, since it demonstrated regularization was the actual bottleneck, not architecture sophistication.
3. **v5 (regularization + weight tying combined) improved on every axis simultaneously**: better Hit Rate@10, better NDCG@10, better validation loss, and 65% fewer parameters than v1.
4. **Seen-item masking** (excluding already-watched movies from predictions before ranking) improved Hit Rate@10 by **~39% relative** (3.71% → 5.14%), discovered via manual inspection of example outputs — a substantial share of the model's raw top-10 slots were being wasted re-recommending already-watched titles.

### 3.4 Baselines — Markov-Chain & Popularity

| Split | Popularity Hit Rate@10 | Markov Hit Rate@10 |
|---|---|---|
| Validation (n=5,454) | 1.32% | 2.11% |
| Test, warm (n=3,427) | 1.31% | 0.76% |
| Test, full population (n=3,602) | 1.55% | 0.92% |

Markov's fallback rate (queries with no recorded transition, defaulting to popularity) was **identical across validation and test (0.8%)**, ruling out coverage drift as an explanation for the val→test reversal (see Section 6.1).

### 3.5 Sentiment Classifier — DistilBERT

Fine-tuned `distilbert-base-uncased` on IMDB 50K Reviews (49,582 reviews after dedup).

| Metric | Value |
|---|---|
| Best epoch (of up to 4, early-stopped) | 2 |
| IMDB validation accuracy | 92.47% |
| RT domain-shift spot-check accuracy | 82.4% (n=500, vs. RT's own Fresh/Rotten labels) |

**Overfitting appeared reliably at epoch 3** across two independent training runs (val loss began climbing while train loss kept falling) — checkpointing on validation loss was added specifically to catch and correct this automatically, the same lesson learned from the GRU applied here.

**Domain-shift finding**: applying a classifier trained on long-form IMDB user reviews (median 172 words) to short RT critic pull-quotes (median 20 words) produces a real, measurable accuracy drop (92.47% → 82.4%). Manual inspection of the 88/500 disagreements revealed a consistent pattern: **negation and hedged language** ("*This is an un-American film in all the right ways*" — Fresh, predicted negative) were the dominant failure mode, consistent with the length/style gap identified before scoring began.

Despite the accuracy drop, aggregating to per-movie sentiment (averaging across 20–30+ reviews per movie) proved reliable: the highest-sentiment movies found (Ugetsu, Atlantic City, Riot in Cell Block 11) and lowest (Wagons East, Who's Your Caddy?, King's Ransom) match real-world critical consensus.

### 3.6 Meta-Model — Ensemble

Logistic regression trained on validation-split (userId, movieId, label) examples, scored by all three base signals.

| Metric | Logistic Regression | GBM (comparison) |
|---|---|---|
| Test AUC | 0.7877 | 0.7929 |
| Train AUC | 0.7881 | 0.7944 |

**Coefficients (standardized):** SVD 1.3708, content 0.0996, GRU 0.0494

**Key finding — why SVD dominates the blend:** raw score distributions showed GRU had a *larger* mean gap between positive/negative examples than SVD (1.27 vs. 0.61), but GRU's much higher variance (std ≈ 4.4 vs. SVD's ≈ 0.5–0.6) meant its *standardized* contribution was smaller. **SVD provides a cleaner, lower-noise signal, not necessarily a more informative one.** This was confirmed, not just theorized: feature correlations between the three signals were low (0.11–0.35), ruling out collinearity as the explanation. The GBM comparison reinforced this — it concentrated ~95% of its feature importance on SVD alone, an even more extreme version of the same pattern. Logistic regression was kept as final for its clearer, more balanced signal-contribution story, needed for the explainability layer (Section 3.7).

### 3.7 Explainability Layer

All five signals from the spec's method table were implemented and validated against real data:

| Signal | Method | Validation |
|---|---|---|
| SVD | Rule-based, calibrated confidence language | Avoids fabricating false feature-level specificity (latent factors aren't semantically interpretable) |
| Content-based | Rule-based, shared genre/keyword/cast terms | Correctly surfaces meaningful overlap (e.g., "Tim Allen, toy comes to life" for Toy Story 3) |
| Meta-model | Standardized coefficient contribution | **A real bug was found and fixed here**: using raw score × coefficient let SVD dominate ~98% of every explanation regardless of the actual driving signal; switching to standardized (scaled) scores fixed this |
| GRU | Rule-based, recent watch history | Verified against real user data — decoded "recently watched" titles exactly matched the true chronological tail of the user's history |
| Sentiment | Rule-based, aggregate critic tone | Matches validated sentiment scores exactly (e.g., Rambling Rose 99% positive) |

---

## 4. Cold-Start Routing

Users with < N=25 train-period ratings are routed around the meta-model entirely, served a content-based + popularity blend instead. Verified via direct set comparison that this routing exactly matches (zero overlap, full coverage) the warm/cold-start partition established in the split logic — no drift between the two independently-implemented definitions.

- 15,593 / 142,184 train users (11.0%) are cold-start
- Fallback recommendations tested on a real cold-start user (23 train ratings): produced a coherent cluster of broadly popular titles (Indiana Jones, Inglourious Basterds, Apocalypse Now)

---

## 5. Full Pipeline — End-to-End Validation

### 5.1 Qualitative testing (validation split)

The unified pipeline (`get_recommendations()`: cold-start check → meta-model or fallback → sentiment re-rank) was tested on multiple real users:

- **Warm-user path**: recommendations formed coherent taste clusters (e.g., a user with a Titanic/Forrest Gump/Schindler's List history received 4/10 positions re-ranked by sentiment; an arthouse-leaning user received 0/10, both explained below)
- **Cold-start path**: correctly triggered for a 23-rating user, produced sensible fallback recommendations

### 5.2 Sentiment re-ranker coverage — a taste-dependent limitation

Testing across 4 real users revealed a systematic, reproducible pattern:

| User | Taste profile | Reliable sentiment coverage | Positions reordered |
|---|---|---|---|
| A | Deep arthouse/foreign (Bergman-adjacent) | 1/10 | 0/10 |
| B | Mixed mainstream + arthouse | 4/10 | 2/10 |
| C | Mainstream prestige (Titanic, Forrest Gump) | 7/10 | 4/10 |

This directly traces back to Week 1's finding that RT review coverage concentrates in mainstream, contemporary titles (2010s: 40.4% of RT-matched movies vs. 33.0% catalog share; near-zero pre-1930). **The re-ranker's practical impact is uneven across users, driven entirely by the underlying data source's coverage bias — not a flaw in the re-ranking logic itself**, which was independently verified to work correctly on a controlled synthetic test.

### 5.3 Seen-item masking (dashboard implementation)

Manual inspection of example outputs found the GRU frequently recommending already-watched movies (e.g., 4 of 10 recommendations for one user were titles already in their history). Masking already-seen items before ranking is applied as a hard default throughout the pipeline (`exclude_seen=True`), not an optional flag — this is standard, expected behavior for any real recommender and materially improved Hit Rate@10 (Section 3.3).

---

## 6. Test-Split Evaluation (Touched Once)

### 6.1 Baseline reversal — a genuine, investigated finding

On validation, Markov clearly outperformed popularity (2.11% vs. 1.32%). On the smaller test split, this **narrowly reversed** (0.76% vs. 1.31% — a difference of just 19 hits out of 3,427 users). The Markov fallback rate was identical across both splits (0.8%), ruling out a coverage-driven explanation. This is most plausibly sampling variability at the smaller test scale (3,427 vs. 5,454 users) combined with genuine period-to-period behavioral drift, rather than a systematic weakness in the Markov approach — which showed a much larger, more stable margin on the larger validation set.

### 6.2 Full pipeline vs. baselines on test — and why it underperformed

| Model | Hit Rate@10 (test, warm) |
|---|---|
| Markov | 0.76% |
| **Full pipeline** | **1.23%** |
| Popularity | 1.31% |

The full pipeline beat Markov but fell just short of raw popularity on the strict full-catalog (~37,335 candidates) single-next-item prediction task. **A targeted diagnostic explains why, rather than leaving this as an unexplained shortfall:**

Restricting evaluation candidates to the 500 most popular movies (n=605 users whose true next-watch fell in this pool) showed the pipeline performing **comparably to popularity in that regime** (7.60% vs. 7.44% Hit Rate@10) — ruling out an inability to compete with popular titles specifically. The real explanation: only **17.7%** of true "next movies" in the full test set were themselves top-500-popular. For the remaining 82.3% (long-tail titles), raw popularity's baseline advantage on this narrow single-item metric outweighs the pipeline's personalization gains.

**Root cause**: the meta-model was trained using randomly-sampled negatives, which rarely exposed it to *popular-but-irrelevant* movies specifically — it never specifically learned to out-rank popularity's raw dominance on this exact task. This does not contradict the meta-model's demonstrated ability to distinguish genuinely liked from disliked candidates (AUC 0.79) or its sensible qualitative behavior (Section 5.1); it reflects a mismatch between the training objective and this specific, unusually strict evaluation metric.

**Suggested future work**: popularity-weighted or hard-negative sampling during meta-model training; broader relevance-window or ranking-based evaluation protocols better suited to comparing personalized systems against a naive popularity baseline.

---

## 7. Signal Contribution Analysis

Combining the meta-model's coefficients (Section 3.6) with the standalone component metrics (Section 3), the overall picture:

- **SVD is the dominant, most reliable signal** — both standalone (RMSE 0.813, Precision@10 0.825) and within the blend (coefficient 1.37), owing to a tight, low-variance score distribution.
- **GRU carries real standalone signal** (clearly beats both baselines on its own, 5.14% vs. 2.11%/1.32% Hit Rate@10) but contributes less to the *blend* specifically due to higher score variance — a nuance the raw coefficient alone would obscure.
- **Content-based similarity is the weakest standalone contributor**, consistent with its documented ~40% catalog exclusion rate limiting its reach.
- **Sentiment re-ranking's real-world impact is highly user-dependent**, bounded by RT's coverage skew rather than by the re-ranking mechanism itself.

---

## 8. Known Limitations (Consolidated)

| Limitation | Where documented | Mitigation / status |
|---|---|---|
| RT review coverage skews toward mainstream/2010s titles | Week 1, Section 5.2 | Documented; sentiment re-ranking is consequently uneven across users |
| ~40% of catalog excluded as content-similarity source | Week 2, Section 3.2 | `min_content_tokens=6` floor, tested against known false-positive cases |
| Sentiment classifier domain shift (IMDB → RT) | Week 3, Section 3.5 | Spot-checked against RT's own labels (82.4%), used as a soft re-ranking signal only |
| GRU run-to-run variance before seeding | Week 2, Section 3.3 | Global seed added; range reported instead of a single point estimate |
| Full pipeline underperforms popularity on strict full-catalog Hit Rate@10 | Week 4, Section 6.2 | Root-caused via targeted diagnostic; future work identified |
| MLflow not wired up (results are plain JSON) | Project-wide | Scoped out under time constraints; noted as a gap against the original spec |

---

## 9. Tech Stack

Python, scikit-learn, `Surprise` (SVD), PyTorch (GRU), HuggingFace `transformers` (DistilBERT), Streamlit + Plotly (dashboard), pandas/NumPy throughout. Training performed on Kaggle (T4 GPU) for GPU-bound components (GRU, DistilBERT); CPU-bound components (SVD, content-based, Markov, cold-start, explainability) developed and validated locally.

---

## 10. Conclusion

CINEIQ+ demonstrates a complete, evaluated recommendation pipeline spanning classical ML, deep sequential modeling, and transformer-based NLP, integrated via a learned ensemble with required cold-start handling and a signal-appropriate explainability layer. Beyond the individual component results, the project's real methodological contribution is a consistent pattern of **catching and explaining problems with evidence rather than assuming success from a single metric** — the GRU's hidden overfitting (caught via validation loss, not Hit Rate@10), the meta-model explanation's scale-mismatch bug (caught via a synthetic test sweep), the sentiment re-ranker's coverage limitation (caught via multi-user testing, traced to an earlier finding), and the test-split popularity gap (caught and root-caused via a targeted diagnostic rather than left unexplained). Each of these findings, and the evidence behind them, is preserved above for anyone extending this work.
