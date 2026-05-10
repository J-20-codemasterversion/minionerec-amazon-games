# Results and engineering trade-offs

## Headline numbers

| Method | HR@10 | NDCG@10 | Test set | Notes |
|---|---|---|---|---|
| Random baseline | ~0.0006 | ~0.0003 | full | sanity floor (10 / 16 572 items) |
| **Classical cascaded recsys (mine)** | **0.045** | **0.024** | 200-user demo | full pipeline: ItemCF/UserCF/Swing recall → LightGBM coarse → FM fine → re-rank |
| Generative (this repo, freeze_LLM, ~5 % of 1 epoch) | 0.002 | 0.001 | 1 000-row test subset | early-stopped due to wall-clock budget |
| *Reference: paper, Beauty dataset, full setup, 8× A100, +ORPO* | *~0.13* | *~0.07* | *full* | *not directly comparable: different dataset and full schedule* |

**The honest read of these numbers**: with the time/hardware budget I had,
the generative approach did **not** beat my classical baseline on Amazon
Video Games. That is the expected outcome for ~5 % of one epoch under
freeze_LLM, and I am reporting it as such rather than cherry-picking.

What the experiment **did** show:
- The **full pipeline runs end-to-end** on commodity hardware (single V100 32 GB).
- The **freeze_LLM SFT path is no longer broken** (see [debugging_log.md](debugging_log.md)).
- I have a **clear next-step plan** with falsifiable predictions (below).

---

## Engineering trade-offs (paper config vs mine)

| Knob | Paper | Mine | Why I diverged |
|---|---|---|---|
| GPUs | 8× A100 | 1× V100 32 GB | hardware constraint |
| Trainable params | 100 % (full SFT) | ~15 % (freeze_LLM: embedding new rows + lm_head) | activation memory + avoid catastrophic forgetting on small data |
| `cutoff_len` | 1024 | 256 | OOM otherwise; ~95th-percentile sequence is < 256 tokens for Games |
| `micro_batch_size` | 4 | 24 | freeze_LLM saves activation memory ⇒ larger micro-batch fits |
| `batch_size` (effective) | 32 (4 × 8 cards) | 128 (24 × ~5 grad-accum) | want a stable signal under fewer optimizer steps |
| Epochs | 3 (+ ORPO post-training) | 1 (no ORPO) | wall-clock budget |
| Train users | 45 884 (all) | 5 000 random subset for fast iter | wall-clock budget |
| Test set | full | 1 000-row subset | inference at `num_beams=20` is the slow part |

---

## Where the bottleneck actually is

After running C-mode (5 000 users, 1 epoch, freeze_LLM) I observed that
training **plateaus at loss ≈ 6.8 after about 100 optimizer steps**. The
gradient-norm and learning-rate schedule both look healthy at that point;
the model is simply data-limited and time-limited.

Concretely, at step ≈ 100 of an ~860-step epoch:
- 768 newly-added SID tokens have only seen a small fraction of the training
  signal,
- the embedding rows for them have not yet aligned to the RQ-Kmeans+
  codebook centers,
- so beam search's logits over the SID vocabulary are still nearly uniform,
  giving HR@10 close to random.

This is consistent with what one would expect when the **only** trainable
parameters are 768 fresh embedding rows and the tied lm_head, and the
training budget is ~5 % of one epoch.

---

## What I would do next (in priority order)

1. **Full epoch on full user set** (~6 h on V100). Pre-registered prediction:
   HR@10 lands in the 0.02–0.04 range. If it lands above 0.04, freeze_LLM
   is more sample-efficient than I estimated; if below 0.02, the
   newly-added embeddings need a higher learning rate or a longer warmup.

2. **ORPO post-training** as in the paper.

3. **SASRec baseline** for a fairer same-architecture comparison.

4. **Profile collision behaviour**: which item categories collide most in
   the 6.25 % of items that share an `<a><b><c>` triple, and does the
   `<d_*>` 4th-level tie-breaker hurt downstream beam search quality?

5. **Replace freeze_LLM with LoRA** on attention layers and compare:
   does training adapters in attention beat training only the new
   embedding rows, given the same trainable-parameter budget? (My intuition
   says no, because the bottleneck is teaching Qwen the *meaning* of
   the new SID tokens, not the *flow* between tokens — but this should be
   checked.)

---

## Files

- `results/baseline_demo_eval.json` — the cascaded baseline numbers.
- `results/games_eval_C_summary.txt` (on COS, not in this repo)  — the
  generative-model evaluation summary including the per-K HR/NDCG breakdown.
