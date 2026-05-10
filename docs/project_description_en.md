# End-to-End Recommendation System on Amazon Review Data

## Project Overview

Built a **production-grade, four-stage recommendation pipeline** from scratch on the Amazon 5-core review dataset (Video Games + Electronics), covering the full lifecycle: data engineering, multi-channel recall, coarse ranking, fine ranking, re-ranking, offline evaluation, and live demo deployment on Tencent Cloud CVM with a Gradio web interface.

**Scale**: 10,000 active users × 47,160 items × 276,833 interactions.

---

## System Architecture

```
Full Item Catalog (47,160)
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Multi-Channel Recall  │  47,160 → 1,000 candidates
│  7 parallel retrieval channels  │  Recall@1000 = 28%
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│  Stage 2: Coarse Ranking        │  1,000 → 100
│  LightGBM binary classification │  6-dim features, pointwise
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│  Stage 3: Fine Ranking          │  100 → 30
│  FM + BPR pairwise ranking      │  16-dim features, pairwise
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│  Stage 4: Re-Ranking            │  30 → 10
│  Greedy diversification rules   │  Brand dedup + freshness + multi-source bonus
└─────────────────────────────────┘
```

---

## Stage 1: Multi-Channel Recall (47,160 → 1,000)

Designed and implemented **7 parallel retrieval channels**, each capturing a different signal of user preference:

| Channel | Algorithm | Weight | Key Idea |
|---------|-----------|--------|----------|
| **ItemCF** | IUF-weighted cosine similarity on item co-occurrence matrix | 0.30 | Users who bought A also bought B; IUF (Inverse User Frequency) penalizes power users to prevent popular-item bias |
| **UserCF** | Jaccard similarity via inverted index (item → user) | 0.20 | Similar users' purchases as candidates; inverted index avoids O(n²) user-pair enumeration |
| **Swing** | Swing algorithm (Alibaba, 2020) | 0.40 | Addresses ItemCF's popularity bias: co-purchases by users with *low* overlap are more informative. Formula: `swing(i,j) = Σ_{u,v} 1/(α + |I_u ∩ I_v|)` |
| **Hot** | Time-decayed weighted popularity | 0.10 | Recency-aware global/per-category hot lists; linear time decay within a 365-day window |
| **Content** | Inverted index on brand / category / title keywords | 0.25 | Content-based filtering using structured metadata attributes |
| **Also-Buy/View** | 2-hop graph traversal on Amazon's co-purchase graph | 0.50 | Leverages Amazon's precomputed item-association graph with 2-hop expansion for coverage |
| **Category-Hot** | User's top-3 preferred categories → category-specific hot items | 0.15 | Personalized popularity within the user's interest categories |

### Fusion Strategy
- Each channel's scores are **min-max normalized** to [0, 1], then **weighted-summed**.
- **Multi-source hit bonus**: Items recalled by multiple channels receive a multiplicative boost: `score × (1 + 0.15 × num_sources)`, rewarding consensus across heterogeneous signals.
- Final output: Top-1,000 candidates per user.

### Key Implementation Details
- **ItemCF**: Co-occurrence matrix built with IUF weighting `w = 1/log(1+n)` where n = user's interaction count. Users with >500 or <2 interactions filtered. Top-50 similar items retained per item via `heapq.nlargest`.
- **UserCF**: Inverted index construction for efficient co-occurrence counting. Items with >500 or <2 buyers filtered to bound computation. Top-30 similar users per user via Jaccard.
- **Swing**: α=0.5 smoothing factor. Max 50 common users per item pair (truncation to avoid O(n²) user-pair explosion). Items filtered to [5, 500] buyer count range. Up to 30,000 items selected for index construction.
- **Also-Buy/View**: 2-hop graph expansion — top-50 first-hop items expanded to 10 second-hop neighbors each, with 0.3× score decay.

---

## Stage 2: Coarse Ranking — LightGBM Binary Classification (1,000 → 100)

### Design Rationale
Coarse ranking requires **speed over precision** — we need to score 1,000 candidates quickly. LightGBM provides:
- Native categorical feature support and automatic feature interaction via gradient boosted trees
- C++ multi-threaded inference — millisecond-level batch scoring
- Built-in L1/L2 regularization preventing overfitting on sparse interaction data

### Feature Engineering (6 dimensions)
Deliberately **lightweight features only** — no recall scores (to avoid train-serve skew / label leakage):

| Feature | Description | Signal Type |
|---------|-------------|-------------|
| `co_interact` | Number of user's neighbor-users who interacted with the candidate item | Collaborative filtering |
| `jaccard` | Jaccard coefficient between user's item set and candidate's buyer set | Interest overlap |
| `item_rating_cnt` | Number of ratings the item received (capped at 100) | Item popularity |
| `item_avg_rating` | Mean rating of the item | Item quality |
| `user_avg_rating` | User's mean rating across all interactions | User rating tendency |
| `item_pop_rank` | Log-transformed global popularity rank of the item | Relative popularity |

### Training Setup
- **Positive samples**: User's actual interactions (label=1)
- **Negative samples**: 4 randomly sampled non-interacted items per positive (label=0)
- **Total**: ~250K samples (50K pos + 200K neg), 80/20 train/val split
- **Hyperparameters**: max_depth=5, learning_rate=0.05, num_boost_round=500, early_stopping_rounds=15, subsample=0.8, colsample_bytree=0.8, min_child_samples=50, L2=1.0, L1=0.1
- **Early stopping** typically converges at ~167 trees

### Inference
```
final_score = LightGBM_probability + 0.3 × normalized_recall_score
```
Recall score blending preserves the recall stage's ordering signal while adding the model's pointwise prediction.

---

## Stage 3: Fine Ranking — Factorization Machine with BPR Loss (100 → 30)

### Design Rationale
- **Why FM over LightGBM for fine ranking?** FM's latent vectors learn arbitrary **second-order feature interactions** (e.g., "user who prefers Sony × item in $50–100 price range"), which is critical when distinguishing among already-similar candidates.
- **Why BPR (pairwise) over binary cross-entropy (pointwise)?** At this stage, we don't care about absolute click probability — we care about **relative ordering**. BPR directly optimizes `P(user prefers item_a over item_b)`, which aligns with the ranking objective.

### Feature Engineering (16 dimensions — expanded from coarse ranking)

| Category | Features | Dims | New in Fine Ranking |
|----------|----------|------|---------------------|
| User | avg_rating, rating_cnt, rating_std | 3 | rating_std added |
| Item | price, log_rank, n_also_buy, n_also_view, avg_rating, rating_cnt | 6 | price, rank, co-purchase graph degree |
| Cross | co_interact, jaccard, category_pref_match, brand_pref_match | 4 | category/brand preference alignment |
| Recall Scores | itemcf_score, swing_score, usercf_score | 3 | Real-valued recall scores from each channel |

### FM Model Formulation
```
score(u, i) = w₀ + w_user[u] + w_item[i] + w · x
              + 0.5 × [‖v_u + v_i + V·x‖² − (v_u² + v_i² + V²·x²)]
```
- `w₀`: global bias
- `w_user[u]`, `w_item[i]`: per-user / per-item first-order bias
- `w · x`: linear combination of 16 dense features
- Interaction term: O(nK) second-order feature crosses via latent vectors (K=8)

### BPR Training
- **Training data**: 200,000 (user, positive_item, negative_item) triplets
- **Loss**: `L = −log σ(s_pos − s_neg)` (BPR loss)
- **Optimizer**: Mini-batch SGD, batch_size=512
- **Regularization**: L2 on weights (λ_w=0.01) and latent vectors (λ_v=0.01)
- **Gradient clipping**: ±5.0 to prevent exploding gradients
- **Feature normalization**: z-score normalization (mean/std computed on training set)
- **Early stopping**: patience=5 on validation BPR-AUC
- **Epochs**: up to 30 (typically converges in ~6 epochs)

### Inference
```
final_score = FM_score + 0.3 × coarse_rank_score
```

---

## Stage 4: Re-Ranking — Greedy Diversification (30 → 10)

### Design Rationale
Re-ranking is a **business logic layer**, not a modeling layer. The goal shifts from relevance to **user experience**: avoiding monotonous recommendations, giving exposure to fresh items, and rewarding high-confidence candidates.

### Algorithm: Greedy Sequential Selection
At each step, select the candidate maximizing:

```
final_score = FM_score + brand_penalty + freshness_bonus + multi_source_bonus
```

| Rule | Condition | Adjustment |
|------|-----------|------------|
| Brand deduplication | Brand already in the selected list | −0.3 |
| Freshness bonus | Item has <10 ratings (cold/new item) | +0.05 |
| Multi-source bonus | Recalled by N channels simultaneously | +0.05 × N |

After each selection, the selected item's brand is added to the seen set, dynamically penalizing subsequent same-brand candidates. This is a **greedy MMR-like** (Maximal Marginal Relevance) approach.

**Result**: Average brand diversity (unique/total) ≈ 0.8+.

---

## Data Engineering

### Dataset Selection Rationale
- Used **Amazon 5-core subsets** (Video Games + Electronics) rather than the full 82GB All-Amazon dataset
- **Why**: The 5-core guarantee (≥5 interactions per user and item) ensures sufficient collaborative filtering signal. Scanning only 4% of the full dataset would truncate most users to ~1.9 interactions, destroying data quality.
- **Preprocessing**: Selected top-10,000 most active users (≥10 interactions each), capped at 100 interactions per user (sorted by timestamp), yielding 276,833 interactions

### Train/Test Split
- **Leave-Last-One-Out**: For each user, the chronologically last interaction is held out for testing; all prior interactions form the training set.
- Ensures temporal integrity — no future information leakage.

### Metadata Integration
- Loaded 47,160 items' metadata (title, brand, category, price, also_buy, also_view) from Amazon's meta JSON files
- ~95% metadata coverage rate
- `also_buy`/`also_view` edges retained for graph-based recall

---

## Evaluation

### Protocol
- **Leave-Last-One-Out** evaluation on 200 test users
- Metrics computed independently at each pipeline stage

### Results Summary

| Stage | Candidates | HitRate@10 | AUC | Notes |
|-------|------------|------------|-----|-------|
| Recall | 47,160 → 1,000 | — | — | Recall@1000 = 28% |
| Coarse (LightGBM) | 1,000 → 100 | 4.00% | 0.522 | 100% hit retention from recall |
| Fine (FM-BPR) | 100 → 30 | 2.00% | 0.505 | BPR validation AUC = 0.999 |
| Re-rank (Rules) | 30 → 10 | 2.00% | 0.506 | Brand diversity ≈ 0.8 |

### Stage Retention Analysis
- Recall → Coarse: **100%** hit retention (coarse ranking preserves all recall hits)
- Coarse → Fine: **50%** hit retention
- Fine → Re-rank: **100%** hit retention

### Health Checks (Built into Pipeline)
- Data: interaction density, rating distribution, metadata coverage
- Recall: per-channel reachability diagnosis, empty recall rate, source contribution distribution
- Coarse: train/val AUC gap (overfitting check), feature importance via gain
- Fine: BPR loss convergence, weight NaN/Inf check, per-feature weight magnitudes
- Re-rank: brand diversity score

---

## Deployment

- **Offline pipeline** (`recommendation_pipeline_v3.py`): Trains all models, serializes the full engine state (model weights, similarity indices, feature statistics, recall objects) into a single `demo_state.pkl` file (~hundreds of MB)
- **Online serving** (`demo_app.py`): Loads `demo_state.pkl`, reconstructs the `RecommendEngine` class that replicates the exact same recall → coarse → fine → rerank logic, and launches a **Gradio** web interface on port 7860
- **Infrastructure**: Tencent Cloud CVM, publicly accessible via security group rules opening port 7860
- **Inference latency**: Full 4-stage pipeline per user request completes in **~seconds** (dominated by recall computation over 47K items)

---

## Known Limitations & Future Improvements

| Limitation | Root Cause | Proposed Solution |
|------------|------------|-------------------|
| Recall ceiling at 28% | Limited to co-occurrence & graph signals; long-tail items unreachable | Add **embedding-based recall** (Item2Vec, ALS) for semantic generalization |
| Fine-ranking AUC ≈ 0.5 on pipeline candidates | 16-dim handcrafted features lack discriminative power among pre-filtered similar items | Upgrade to **DeepFM / xDeepFM** with richer features (review text TF-IDF, price-range preferences) |
| Random negative sampling | Train/eval distribution mismatch — random negatives are too easy | **Hard negative mining** from recall candidates |
| No real-time features | Static features only; no session-level signals | Incorporate real-time click/view sequences |

---

## Technical Highlights (Interview Talking Points)

1. **Multi-channel recall with principled fusion**: 7 heterogeneous retrieval channels (collaborative filtering, content-based, graph-based, popularity-based) unified through normalized weighted scoring with multi-source consensus bonuses.

2. **Deliberate model differentiation across stages**: Pointwise LightGBM for coarse ranking (fast, handles feature interactions automatically) vs. Pairwise FM-BPR for fine ranking (optimizes relative ordering directly). This reflects a clear understanding of when pointwise vs. pairwise objectives are appropriate.

3. **Feature leakage prevention**: Coarse ranking deliberately excludes recall scores to avoid train-serve skew. Recall scores are only introduced at the fine-ranking stage where they serve as legitimate features.

4. **Swing algorithm for de-biased collaborative filtering**: Addresses the well-known popularity bias in standard ItemCF — co-purchases by users with small interaction overlap carry stronger signal than co-purchases by power users.

5. **End-to-end pipeline with built-in diagnostics**: Automated health checks at every stage (data quality, recall reachability, overfitting detection, weight sanity, diversity metrics) — not just a model, but a production-mindset system.

6. **Full deployment lifecycle**: From raw JSON data → offline training → model serialization → online serving → public-facing web demo, demonstrating ability to ship a complete ML system rather than just training a model in a notebook.
