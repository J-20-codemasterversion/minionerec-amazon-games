# Generative Recommendation on Amazon Games — End-to-End Reproduction

> **Can a 1.5B LLM learn to *generate* the next product a user wants, token by token, instead of scoring a fixed candidate list?**
> I rebuilt MiniOneRec from raw Amazon reviews to a loadable checkpoint on a single V100, then measured the honest gap against a classical recommender I wrote from scratch.

<p>
<img alt="status" src="https://img.shields.io/badge/status-reproduction%20study-blue">
<img alt="hardware" src="https://img.shields.io/badge/trained%20on-1%C3%97V100%2032GB-success">
<img alt="base model" src="https://img.shields.io/badge/base-Qwen2.5--1.5B-orange">
<img alt="checkpoint" src="https://img.shields.io/badge/checkpoint-ModelScope-8A2BE2">
</p>

Not a leaderboard win, and that is the point: a pipeline that runs end-to-end on one commodity GPU, two real upstream bugs fixed before anything would train, and a baseline strong enough to keep the generative model honest.

**If you only read three things:**
1. 🐛 [The `freeze_LLM` `NameError` nobody upstream ever hit](#1-the-freeze_llm-bug-nobody-upstream-hit).
2. 📉 [The honest results table](#honest-results) — the generative model *loses* to my baseline here, and I say why.
3. 🧩 [The checkpoint that forgot its own vocabulary](#3-a-checkpoint-that-forgot-its-own-vocabulary).

Long log (Chinese): [`docs/reproduction_journey.md`](docs/reproduction_journey.md) · English summary: [`docs/project_description_en.md`](docs/project_description_en.md).

---

## Pipeline at a glance

| Stage | What I did | Outcome |
|---|---|---|
| Preprocess | 5-core + leave-one-out on Amazon Video Games | 16,572 items · 45,884 users |
| Semantic encode | Title+desc → Qwen-2.5-1.5B (1,536-d) | `Games.emb-qwen-td.npy` |
| RQ-Kmeans+ | 3-level residual K-means, collision-aware | 6.25% collision rate |
| Semantic IDs | Each item → `<a_X><b_Y><c_Z>` (+ `<d_*>` tie-break) | 95.4% unique SIDs |
| Freeze-LLM SFT | Add 768 SID tokens, freeze trunk, train embeds + `lm_head` (~15% params) | loss 8.4 → 6.8, early stop |
| Inference | Constrained beam search over legal SID triples | HR@10 / NDCG@10 |

A **classical cascaded baseline** (recall + coarse-rank + fine-rank + re-rank) on the same data lives in [`baseline/`](baseline/).

---

## Three things worth your time

### 1. The `freeze_LLM` bug nobody upstream hit

MiniOneRec defines `original_vocab_size` only inside the `train_from_scratch=True` branch, but the `freeze_LLM` path references it unconditionally — so `(train_from_scratch=False, freeze_LLM=True)` crashes with `NameError`. That combination is exactly what you want on a single V100 (keep Qwen's pretrained weights, train only the new SID embeddings + `lm_head`), and it is the path the default config never exercises. One-line fix, but it meant reading enough of the code to understand how `resize_token_embeddings` and slice-level `requires_grad=False` interact. The buggy original is kept as `sft.py.original_buggy` for an auditable diff.

### 2. Trade-offs under one V100

| Knob | Paper | Mine | Why |
|---|---|---|---|
| GPUs | 8× A100 | 1× V100 32GB | hardware |
| `cutoff_len` | 1024 | 256 | OOM otherwise |
| epochs | 3 | 1 | wall-clock |
| trainable | 100% | ~15% | memory + avoid forgetting |

Detail in [`docs/reproduction_journey.md`](docs/reproduction_journey.md), including the trap where tqdm's "1.04 s/step" was the *micro-batch* step — real per-optimizer-update time was ~50 s after `gradient_accumulation_steps`.

### 3. A checkpoint that forgot its own vocabulary

Loaded back from ModelScope, the model had **152,460** embedding rows but the tokenizer knew only **151,665**: the 795 SID tokens were trained into the weights yet missing from the saved tokenizer, so SID encode/decode silently broke. `Trainer` doesn't write the tokenizer into intermediate checkpoints. The fix must be **order-preserving** (token id `151665 + k` must match the k-th trained row), so reproducing the training-time `sorted(set(...))` over `Games.index.json` restores it exactly — [`tools/rebuild_sid_tokenizer.py`](tools/rebuild_sid_tokenizer.py), with a contiguity assert so a wrong index file fails loudly:

```bash
python tools/rebuild_sid_tokenizer.py --model_dir ./games-sft \
    --index_path ./games-sft/data/Games.index.json --expected_vocab 152460
```

The published checkpoint already has this applied.

---

## Honest results

A reproduction under tight resource constraints, not a competitive benchmark.

| Method | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| Random | ~0.0006 | ~0.0003 | sanity floor |
| **Classical baseline (mine)** | **0.045** | **0.024** | full pipeline, [`baseline/`](baseline/) |
| Generative (this repo) | 0.002 | 0.001 | ~5% of 1 epoch, early-stopped |
| Paper (Beauty, 8×A100, +ORPO) | ~0.13 | ~0.07 | not directly comparable |

The generative model does **not** beat the baseline here — freeze_LLM SFT plateaued at loss ≈ 6.8 after 102 optimizer steps. The win is that the **full pipeline runs end-to-end on commodity hardware**, the **SFT path is now bug-free**, and the **gap is reported honestly**. Next: a full epoch on all users (~6h, expect HR@10 ≈ 0.02–0.04), ORPO post-training, and a SASRec baseline.

---

## Run it

```bash
conda create -n minionerec python=3.10 -y && conda activate minionerec
pip install -r requirements.txt            # torch>=2.0, transformers>=4.40, numpy<2.0

bash rq/text2emb.sh                          # encode items
bash rq/train_constrained.sh                 # RQ-Kmeans+ → Semantic IDs
python convert_dataset.py --dataset Games --data_dir <path>
bash sft_games.sh                            # freeze_LLM SFT (V100-friendly hparams at top)
bash evaluate.sh && python calc.py --path results/<run>.json --item_path <info>

# baseline
cd baseline && python recall_main_v2.py && python lightgbm_ranking_train.py \
    && python fm.py && python recommendation_pipeline_v4.py
```

---

## Pretrained checkpoint (ModelScope)

The ≈2.9 GB SFT checkpoint is archived at [`woshiJ20/MiniOneRec-Amazon-Games-SFT`](https://www.modelscope.cn/models/woshiJ20/MiniOneRec-Amazon-Games-SFT) (currently **private** — make public or add collaborators to share).

```python
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = snapshot_download("woshiJ20/MiniOneRec-Amazon-Games-SFT")
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
assert len(tok) == 152460     # base Qwen vocab + 795 SID tokens
```

Base LLM is `Qwen/Qwen2.5-1.5B-Instruct`; only the SID embeddings + `lm_head` were trained. Inference must restrict beam search to legal `<a_*><b_*><c_*>` triples — see [`evaluate.py`](evaluate.py).

---

## Notes

- Amazon Reviews data is not redistributed here; see [`docs/amazon_recsys_report.md`](docs/amazon_recsys_report.md).
- Checkpoints (3 GB) live on ModelScope + Tencent COS, not in Git.
- What's mine vs upstream: [`docs/MY_CONTRIBUTIONS.md`](docs/MY_CONTRIBUTIONS.md).
