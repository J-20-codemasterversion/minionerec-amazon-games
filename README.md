# Generative Recommendation on Amazon Games — End-to-End Reproduction

> **Can a 1.5B LLM learn to *generate* the next product a user wants, token by token, instead of scoring a fixed candidate list?**
> I rebuilt MiniOneRec from raw Amazon reviews all the way to a loadable checkpoint on a single V100, then measured the honest gap against a classical recommender I wrote from scratch.

<p>
<img alt="status" src="https://img.shields.io/badge/status-reproduction%20study-blue">
<img alt="hardware" src="https://img.shields.io/badge/trained%20on-1%C3%97V100%2032GB-success">
<img alt="base model" src="https://img.shields.io/badge/base-Qwen2.5--1.5B-orange">
<img alt="checkpoint" src="https://img.shields.io/badge/checkpoint-ModelScope-8A2BE2">
</p>

This is not a leaderboard win, and that is the point. It is the messier, more useful artifact: a pipeline that actually runs end-to-end on one commodity GPU, two real bugs in the upstream training code that had to be read and fixed before a single step would train, and a baseline strong enough to keep the generative model honest.

**If you only read three things:**
1. 🐛 [The `freeze_LLM` `NameError` nobody upstream ever hit](#1-patching-a-freeze_llm-bug-the-original-authors-didnt-hit) — and why that exact config is the one you actually want on a single GPU.
2. 📉 [The honest results table](#honest-results) — the generative model *loses* to my classical baseline here, and I explain precisely why instead of hiding it.
3. 🧩 [The checkpoint that forgot its own vocabulary](#5-a-checkpoint-that-forgot-its-own-vocabulary) — 795 trained embedding rows whose tokens the tokenizer had silently dropped, and the order-preserving script that brings them back.

Full reproduction log (Chinese): [`docs/reproduction_journey.md`](docs/reproduction_journey.md) · English summary: [`docs/project_description_en.md`](docs/project_description_en.md).

---

## TL;DR

| Stage | What I did | Outcome |
|---|---|---|
| 1. Data preprocessing | 5-core filter + leave-one-out split on Amazon Video Games (16 572 items, 45 884 users) | clean train/valid/test |
| 2. Semantic encoding | Encode each item title+description with Qwen-2.5-1.5B (1 536-d) | `Games.emb-qwen-td.npy` (101 MB) |
| 3. RQ-Kmeans+ quantization | Three-level residual K-means with collision-aware loss | **6.25% collision rate**, codebook saved |
| 4. Semantic ID generation | Map each item to `<a_X><b_Y><c_Z>` token triple (+ optional `<d_*>` tie-breaker) | **95.4% items get unique SID** |
| 5. Dataset format conversion | Build (history-SID-sequence → next-SID) training pairs | 103 023 SFT samples |
| 6. **Freeze-LLM SFT** | Add 768 new SID tokens to Qwen tokenizer, freeze the 28-layer Transformer trunk, train only embedding (768 new rows) + lm\_head (~15% trainable params) | training stops early due to single-GPU time budget; loss 8.4 → 6.8 |
| 7. Constrained beam-search inference | Beam search restricted to legal `<a><b><c>` SID triples on a 1k-test subset | HR@10 / NDCG@10 reported |

I also implemented a **classical cascaded baseline** (recall + coarse-rank + fine-rank + re-rank) on the same dataset for direct comparison — see [`baseline/`](baseline/).

---

## Repository layout

```
.
├── sft.py                       # Patched: fixes a NameError in the freeze_LLM branch
├── evaluate.py                  # Constrained beam-search inference
├── calc.py                      # HR@K / NDCG@K computation
├── convert_dataset.py           # Build SID-format training data
├── rq/                          # RQ-Kmeans+ training code
├── baseline/                    # Classical cascaded recsys (my own implementation)
│   ├── recall_itemcf.py / recall_swing.py / recall_usercf.py / recall_hot.py / recall_fusion.py
│   ├── recall_main_v2.py        # Multi-channel recall pipeline
│   ├── lightgbm_ranking_train.py
│   ├── fm.py                    # FM fine-ranking model
│   ├── feature_regression.py
│   └── recommendation_pipeline_v3.py / v4.py / full.py    # End-to-end orchestration
├── docs/
│   ├── reproduction_journey.md          # Full reproduction log (Chinese, primary)
│   ├── project_description_en.md        # Concise English project summary
│   ├── cascaded_pipeline_report.md      # Baseline pipeline analysis
│   ├── pipeline_overview.md
│   ├── amazon_recsys_report.md
│   ├── recsys_plan.md
│   └── demo_setup.md                    # Cloud GPU demo deployment notes
├── results/
│   └── baseline_demo_eval.json          # Baseline HR@10/NDCG@10 numbers
├── data/                                # Sample Amazon Games splits (small files only)
└── requirements.txt
```

---

## What was hard / what I learned

### 1. Patching a freeze\_LLM bug the original authors didn't hit

The MiniOneRec training script defines `original_vocab_size` only inside the `train_from_scratch=True` branch. The downstream `freeze_LLM` code path then unconditionally references that variable to slice the embedding matrix:

```python
# fails with NameError on (train_from_scratch=False, freeze_LLM=True)
for p in model.embed_tokens.parameters():
    p[:original_vocab_size].requires_grad = False
```

The combination `(train_from_scratch=False, freeze_LLM=True)` is exactly the configuration you want on a single V100 (preserve Qwen's pretrained knowledge, only train the new SID-token embeddings + `lm_head`). It is also the path the authors' default config never exercises. I read the source, located the missing definition, and patched it with a single line:

```python
original_vocab_size = len(tokenizer)
tokenizer.add_tokens(new_sid_tokens)
```

This is a small fix, but it required reading enough of the codebase to understand:
- how `resize_token_embeddings` interacts with newly added tokens,
- which slice of the embedding matrix corresponds to "original Qwen vocab" vs "newly added SID tokens",
- why `requires_grad=False` on a slice is the right freeze mechanism here.

### 2. Engineering trade-offs under a single V100 32GB

Compared to the paper's 8× A100 setup, every choice is a trade-off:

| Knob | Paper | Mine | Why |
|---|---|---|---|
| GPUs | 8× A100 | 1× V100 32GB | hardware constraint |
| `cutoff_len` | 1024 | 256 | OOM otherwise |
| `micro_batch_size` | 4 (×8 cards) | 24 | freeze\_LLM saves activation memory |
| `num_epochs` | 3 | 1 | wall-clock budget |
| Trainable params | 100% | ~15% (freeze\_LLM) | memory + avoid catastrophic forgetting |
| Training data | full (45k users) | 5k user subset for fast iteration | wall-clock budget |

I documented these in [`docs/reproduction_journey.md`](docs/reproduction_journey.md) (in Chinese), including the surprises — e.g., I initially saw "1.04 s/step" in tqdm and thought training would finish in 30 min, but that was the *micro-batch* step, not the optimizer step. After accounting for `gradient_accumulation_steps`, real wall-clock per optimizer update was closer to 50 s.

### 3. Constrained beam search and token-level evaluation

Inference uses beam search restricted to legal SID triples (`<a_*>` then `<b_*>` then `<c_*>`). Mapping the generated SID back to a real item via `Games.index.json`, then computing HR@10 / NDCG@10, completes the loop. With only ~5% of one epoch of training under freeze\_LLM, the resulting HR@10 is **far below the cascaded baseline** — which is exactly the expected outcome and is reported honestly in [`results/`](results/) rather than cherry-picked.

### 4. Classical baseline as honest reference

To make the comparison meaningful I built the cascaded baseline ([`baseline/`](baseline/)) myself on the same Amazon Games data:
- **Recall**: ItemCF, UserCF, Swing, popularity — fused via [`recall_fusion.py`](baseline/recall_fusion.py)
- **Coarse rank**: LightGBM ([`lightgbm_ranking_train.py`](baseline/lightgbm_ranking_train.py))
- **Fine rank**: FM ([`fm.py`](baseline/fm.py))
- **Re-rank + evaluation**: [`recommendation_pipeline_v4.py`](baseline/recommendation_pipeline_v4.py)

Reported HR@10 / NDCG@10 are in [`results/baseline_demo_eval.json`](results/baseline_demo_eval.json).

### 5. A checkpoint that forgot its own vocabulary

When I archived the trained checkpoint to ModelScope and loaded it back the way an outside user would, the model came up with **152,460** embedding rows while the tokenizer only knew **151,665** tokens. The 795 Semantic-ID tokens (`<a_*>`, `<b_*>`, `<c_*>`, plus 27 `<d_*>` tie-breakers) were baked into the embedding matrix yet completely missing from the saved tokenizer, so `tokenizer.encode("<a_133>")` quietly fell back to byte-level pieces and SID decoding produced garbage.

The root cause is the one [`docs/debugging_log.md`](docs/debugging_log.md) already flags: `Trainer` does not write the tokenizer into intermediate checkpoints. What makes the fix non-trivial is that it has to be **order-preserving** — token id `151665 + k` must line up with the k-th trained embedding row, or the model silently degrades. Reproducing the training-time `sorted(set(...))` over `Games.index.json` restores that mapping exactly. That is what [`tools/rebuild_sid_tokenizer.py`](tools/rebuild_sid_tokenizer.py) does, with a contiguity assertion so a mismatched index file fails loudly instead of corrupting the model in silence.

```bash
python tools/rebuild_sid_tokenizer.py \
    --model_dir   ./games-sft \
    --index_path  ./games-sft/data/Games.index.json \
    --expected_vocab 152460
```

The checkpoint published on ModelScope already has this fix applied, so the load snippet below works as-is.

---

## How to run

### Environment

```bash
conda create -n minionerec python=3.10 -y
conda activate minionerec
pip install -r requirements.txt
```

Key versions:
- `torch >= 2.0`, `transformers >= 4.40`, `accelerate`, `peft`, `deepspeed`, `fire`
- `numpy < 2.0` (sklearn ABI compatibility)

### Pipeline

```bash
# Stage 2: Qwen embedding (16 572 items × 1 536 dim)
bash rq/text2emb.sh

# Stage 3: RQ-Kmeans+ training
bash rq/train_constrained.sh
bash rq/train_plus.sh

# Stage 4: SID generation
python rq/generate_index.py

# Stage 5: training data
python convert_dataset.py --dataset Games --data_dir <path>

# Stage 6: SFT (freeze_LLM + new SID tokens)
bash sft_games.sh         # see top of file for V100-friendly hyperparameters

# Stage 7: evaluation
bash evaluate.sh
python calc.py --path results/<run>.json --item_path <info file>
```

### Baseline (classical cascaded recsys)

```bash
cd baseline
python recall_main_v2.py
python lightgbm_ranking_train.py
python fm.py
python recommendation_pipeline_v4.py
```

---

## Honest results

This is a **reproduction study under tight resource constraints**, not a competitive benchmark.

| Method | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| Random (16 572 items) | ~0.0006 | ~0.0003 | sanity floor |
| **Classical cascaded baseline (mine)** | **0.045** | **0.024** | full pipeline, mine, see [`baseline/`](baseline/) |
| Generative (this repo, freeze\_LLM, ~5% of 1 epoch, 1k test subset) | 0.002 | 0.001 | early-stopped due to time budget |
| Reference: paper (Beauty, full setup, 8× A100, +ORPO) | ~0.13 | ~0.07 | not directly comparable — different dataset and full training schedule |

The key takeaway is **not** that the generative model outperforms the cascaded one here — it doesn't, because freeze\_LLM SFT plateaued at loss ≈ 6.8 after only 102 optimizer steps under the time budget I had. The takeaway is that the **full pipeline runs end-to-end on commodity hardware**, the **bug-free SFT path now exists**, and the **honest gap analysis** is documented.

---

## What I would do next

- Run a **full** epoch of freeze\_LLM SFT on the full user set (estimated ~6h on V100), then re-evaluate on the full 41 924-row test set. I expect HR@10 to be in the 0.02–0.04 range based on the loss trajectory.
- Add **ORPO** post-training as in the original paper.
- Compare against **SASRec** as an additional sequential-recommendation baseline.
- Profile **collision behavior** of RQ-Kmeans+: which item categories collide most, and does the `<d_*>` 4th-level tie-breaker hurt downstream beam search?

---

## Pretrained checkpoint (ModelScope)

The reproduced Amazon Games SFT checkpoint (≈2.9 GB) is **not** stored in this Git repo. It is archived on ModelScope:

- Model: [`woshiJ20/MiniOneRec-Amazon-Games-SFT`](https://www.modelscope.cn/models/woshiJ20/MiniOneRec-Amazon-Games-SFT)
- Includes: `model.safetensors`, tokenizer/config, `data/Games.index.json` (item → Semantic ID map), and `results/` (constrained beam-search eval).

> Note: the repo is currently **private**. Make it public or add collaborators before sharing, and log in with an access token from <https://modelscope.cn/my/myaccesstoken>.

Download:

```bash
modelscope login --token <YOUR_TOKEN>
modelscope download --model woshiJ20/MiniOneRec-Amazon-Games-SFT --local_dir ./games-sft
```

Load and run:

```python
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = snapshot_download("woshiJ20/MiniOneRec-Amazon-Games-SFT")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,   # use torch.float32 on CPU-only machines
    trust_remote_code=True,
).eval()

# The tokenizer already contains the added SID tokens (<a_*>/<b_*>/<c_*>).
assert len(tokenizer) == 152460
```

The base LLM is `Qwen/Qwen2.5-1.5B-Instruct`; only the new SID-token embeddings + `lm_head` were trained (`freeze_LLM=True`). Constrained beam search must be restricted to legal `<a_*><b_*><c_*>` triples — see [`evaluate.py`](evaluate.py).

---

## Notes

- Amazon Reviews data is not redistributed in this repo; download instructions are in [`docs/amazon_recsys_report.md`](docs/amazon_recsys_report.md).
- Trained checkpoints are large (3 GB); they are archived on **ModelScope** (`woshiJ20/MiniOneRec-Amazon-Games-SFT`, see above) and backed up on Tencent COS — not in this Git repo.
- The reproduction log [`docs/reproduction_journey.md`](docs/reproduction_journey.md) is in Chinese; the English summary lives at [`docs/project_description_en.md`](docs/project_description_en.md).
