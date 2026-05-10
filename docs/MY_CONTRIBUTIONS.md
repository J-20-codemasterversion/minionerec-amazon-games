# What is mine in this repository

This repository contains both my own code and code from the upstream
[MiniOneRec](https://github.com/AkaliKong/MiniOneRec) project. To make this
unambiguous for reviewers, here is the breakdown.

## Things I wrote myself

- **The classical cascaded recommender** in [`baseline/`](../baseline/):
  - Multi-channel recall (`recall_itemcf.py`, `recall_usercf.py`,
    `recall_swing.py`, `recall_hot.py`, `recall_fusion.py`,
    `recall_main_v2.py`, `recall_data_loader.py`)
  - Coarse ranking (`lightgbm_ranking_train.py`)
  - Fine ranking (`fm.py`)
  - Feature engineering (`feature_regression.py`)
  - End-to-end orchestration (`recommendation_pipeline_v3.py` / `v4.py` / `_full.py`)

  These are the comparison baseline I built on the same Amazon Video Games
  data, before I started reproducing the generative approach.

- **All documentation** under [`docs/`](.):
  - [`reproduction_journey.md`](reproduction_journey.md) — full reproduction log (Chinese)
  - [`debugging_log.md`](debugging_log.md) — concrete bugs I found and fixed
  - [`results.md`](results.md) — engineering trade-offs and honest results
  - [`project_description_en.md`](project_description_en.md) — concise English summary
  - [`cascaded_pipeline_report.md`](cascaded_pipeline_report.md), [`pipeline_overview.md`](pipeline_overview.md), [`amazon_recsys_report.md`](amazon_recsys_report.md), [`recsys_plan.md`](recsys_plan.md) — pipeline analyses
  - [`demo_setup.md`](demo_setup.md) — cloud GPU demo deployment notes

- **Patches to the upstream MiniOneRec code** (clearly marked with
  `# PATCH (J-20-codemasterversion):` comments in-line):
  - `sft.py`: define `original_vocab_size = len(tokenizer)` before
    `add_tokens(...)` so the `freeze_LLM=True` branch stops crashing with
    `NameError`. The original buggy file is preserved as
    `sft.py.original_buggy` so the diff is auditable.
  - `sft.py` and `evaluate.py`: add `"Games": "video games"` to the
    hard-coded `category_dict`.

- **Reproduction-specific scripts and configs** (some captured in
  `docs/reproduction_journey.md` and `docs/demo_setup.md`):
  - V100-friendly hyperparameter set (`cutoff_len=256`, `micro_batch_size=24`,
    `batch_size=128`, `num_epochs=1`, freeze_LLM)
  - 5 000-user subset construction
  - 1 000-row test subset for fast evaluation
  - Tokenizer-from-base copy step for mid-training checkpoints

- **Honest reporting** — including the negative result that under the time
  budget I had, generative HR@10 stayed below the classical-baseline HR@10.
  See [`docs/results.md`](results.md).

## Things I did not write

- **The MiniOneRec training pipeline itself** (`sft.py`, `evaluate.py`,
  `calc.py`, `merge.py`, `convert_dataset.py`, `data.py`, `utility.py`,
  `LogitProcessor.py`, `SASRecModules_ori.py`, `sasrec.py`,
  `minionerec_trainer.py`, `rl.py`, `sft_gpr.py`, `convert_dataset_gpr.py`,
  `rl_gpr.py`, `split.py`)
- **The RQ-Kmeans+ training code** under [`rq/`](../rq/)
- **The original README** (`README.md` here is rewritten by me; the upstream
  README structure is preserved in spirit but the content is mine)
- **The `LICENSE`** (carried over from upstream)
- **Config fixtures** under `config/`, `assets/`

These are all from the upstream MiniOneRec repository. Their authors deserve
the credit for the modeling work; my contribution is reproducing it on a
single GPU on a new dataset, debugging the freeze_LLM path that the upstream
default config never exercised, and comparing the result against a classical
cascaded baseline I built myself.

## How to verify my edits to the upstream files

```bash
# show every patch I applied
diff sft.py.original_buggy sft.py
grep -n "PATCH (J-20-codemasterversion):" sft.py evaluate.py
```
