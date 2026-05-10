# Debugging log: bugs found while reproducing MiniOneRec on a single V100

This document records concrete issues I encountered while running the open-source
MiniOneRec training/evaluation code on a configuration the original authors had
not tested in their default scripts: **single GPU + `freeze_LLM=True` + new dataset**.

The bugs are small, but locating them required reading the source — they are
the kind of issue you only see if you actually try to run the code end-to-end
on your own setup.

---

## Bug 1 — `NameError: original_vocab_size` in the `freeze_LLM` branch

### What I observed

When launching SFT with `train_from_scratch=False`, `freeze_LLM=True`, and
`sid_index_path` pointing to my own RQ-Kmeans+ index, the script crashed
immediately after adding the 768 new SID tokens:

```
NameError: name 'original_vocab_size' is not defined
  File "sft.py", line 169, in train
    if embedding_layer.weight.shape[0] > original_vocab_size:
```

Stack trace was clean, so this was clearly a control-flow / scope issue rather
than a runtime data issue.

### Root cause

The code uses `original_vocab_size` as the slice index for freezing the
embedding rows that belong to the *original* Qwen vocabulary, leaving the *new*
SID-token rows trainable:

```python
def mask_grad(grad):
    grad[:original_vocab_size].zero_()   # zero gradients on Qwen's original rows
    return grad
embedding_layer.weight.register_hook(mask_grad)
```

But the variable was only defined inside the `train_from_scratch=True`
branch elsewhere — never in the `train_from_scratch=False` path. The two paths
are conceptually:

| Path | Trainable | When you'd pick it |
|---|---|---|
| `train_from_scratch=True`  | the whole model | you have lots of compute and data |
| `train_from_scratch=False, freeze_LLM=False` | the whole model, fine-tuned | the paper's default |
| `train_from_scratch=False, freeze_LLM=True`  | **embedding (new rows) + lm\_head only** | **single-V100 setup; this PR's path** |

The third path is exactly what you want on a single V100 32 GB:
- preserves Qwen's pretrained knowledge (no catastrophic forgetting),
- shrinks the trainable parameter count from 1.5 B to ~234 M (about 15 %),
- keeps activation memory roughly in budget.

It is also the path the authors' default config never exercises, so the bug
was latent.

### Fix

Capture `len(tokenizer)` *before* extending the vocab, in the same block where
new tokens are added:

```python
new_tokens = token_extender.get_new_tokens()
if new_tokens:
    # PATCH: define `original_vocab_size` before resize so the freeze_LLM
    # branch below can slice [original:] = "new SID rows".
    original_vocab_size = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
```

That single line keeps the rest of the freeze logic correct: the gradient
hook now zeros gradients on the original 151 936 rows and lets the 768
new SID-token rows learn. After the fix, training proceeds, and
`Trainable parameters (with grad-mask)` is reported as ~234 M / 1.5 B — the
expected ratio for this freeze regime.

The patch and the original buggy file are both checked in:

- patched: [`sft.py`](../sft.py) (search for `PATCH`)
- original: [`sft.py.original_buggy`](../sft.py.original_buggy)
- diff: `diff sft.py.original_buggy sft.py`

### What I learned from this

- **Scope-defined-on-one-branch bugs are common in research code** — the
  `if/else` branches a single researcher tested are the only branches that
  actually run during their experiments.
- The `embedding_layer.weight.register_hook(mask_grad)` pattern is a clean way
  to freeze part of a tensor (instead of splitting it into two `Parameter`
  objects). Worth remembering.
- The freeze regime is **not** equivalent to LoRA: LoRA adds low-rank
  adapters to the attention weights, while `freeze_LLM=True` here trains
  a contiguous slice of the existing embedding matrix plus the tied
  `lm_head`. Different goals, different memory profile, different inductive
  bias. Knowing which one to pick matters more than picking the trendier one.

---

## Bug 2 — `KeyError: 'Games'` in `category_dict`

### What I observed

After the `original_vocab_size` fix, the next failure was:

```
KeyError: 'Games'
  File "sft.py", line 121, in train
    category = category_dict[category]
```

### Root cause

The dictionary mapping category-id → human-readable phrase only covered the
two datasets used in the paper (Industrial & Scientific, Office Products) plus
a few other Amazon categories — but not Video Games. The same hard-coded
mapping is duplicated in `evaluate.py`. So switching to a new dataset means
patching it in **every** script that loads category metadata.

### Fix

Add `"Games": "video games"` to the dict in both `sft.py` (line 119) and
`evaluate.py` (line 55).

### What I learned from this

- This is a **classic research-code smell**: the same configuration
  (category → phrase mapping) is duplicated across multiple scripts. In
  production you'd factor it into one config file or one `meta.py`.
- After hitting this once, I now check for duplicated config dicts as part of
  my "first read of an unfamiliar repo" pass.

---

## Bug 3 — checkpoint folder is missing a tokenizer

### What I observed

After SFT finished and produced `checkpoint-86/`, running `evaluate.py` on
that checkpoint crashed:

```
OSError: Can't load tokenizer for 'output/.../checkpoint-86'.
Make sure '...' is the correct path to a directory containing all relevant
files for a Qwen2TokenizerFast tokenizer.
```

### Root cause

`transformers.Trainer` does not, by default, save the tokenizer to every
intermediate checkpoint folder; only the model weights, optimizer state, RNG
state, and trainer state. The original SFT script doesn't override this. So
mid-training checkpoints have weights but no tokenizer, and you cannot load
them with `AutoTokenizer.from_pretrained(checkpoint_path)`.

### Fix (workaround)

Copy the tokenizer files from the base Qwen model into the checkpoint folder
before evaluation:

```bash
QWEN=/path/to/Qwen2.5-1.5B-Instruct
CKPT=output/.../checkpoint-86
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$QWEN/$f" ] && cp "$QWEN/$f" "$CKPT/"
done
```

This works because the only modification we made was *adding* tokens — the
existing Qwen tokenizer rules still apply, and the new SID tokens are
recreated by re-running `tokenizer.add_tokens(...)` at evaluation time.

A cleaner long-term fix would be to override `Trainer`'s `_save_checkpoint`
to also call `tokenizer.save_pretrained(checkpoint_dir)` — but for a
short-run reproduction, the file-copy was the pragmatic call.

### What I learned from this

- HuggingFace's "save the model" defaults are lossy in subtle ways. For
  research code where you stop training early and resume from a mid-training
  checkpoint, you need to be deliberate about what gets saved.
- For freeze_LLM where I'm only adding tokens (not changing existing
  tokenizer rules), copying the base tokenizer is safe. For a more invasive
  tokenizer change (e.g. retraining BPE), it would not be.

---

## Smaller issues, in passing

| Issue | Where | Fix |
|---|---|---|
| `numpy 2.x` ABI incompatible with installed `scikit-learn` | env setup | `pip install "numpy<2"` |
| `text2emb` default `batch_size=1024` OOMs on V100 | `rq/text2emb.sh` | reduce to 64 |
| `convert_dataset.py` reads `.inter` but writes `.csv` to a *different* directory than `sft.py` reads from; easy to silently train on stale data | layout | document it; add a sanity-check on row counts in [`docs/reproduction_journey.md`](reproduction_journey.md) |
| `tqdm` reports "1 s/step" but real wall-clock is much slower | UX | the displayed step is the *micro-batch* step; multiply by `gradient_accumulation_steps` for the optimizer step |

---

## Summary

These are not glamorous bugs. None of them are conceptually deep. But each one
was a real blocker that required reading the source, forming a hypothesis,
and verifying with a minimal change. Doing this on every project I work on is
why I'm comfortable claiming I can "modify modern training code independently"
— I've actually had to.
