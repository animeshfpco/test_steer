# Steering vector extraction + analysis pipeline

Stage 1 (`extract.py`) generates rollouts under each persona's pos/neg system prompts and saves per-layer pooled activations under three pooling strategies. Stage 2 (`analyze.py`) computes diff vectors, AUROC discriminability per layer, and picks candidate layers per strategy. Stage 3 (the α sweep on chosen candidates with full generation + judge eval) is intentionally separate and not in this scaffold.

## Files

```
config.py    # paths, model, persona list, layer ranges — edit values, no logic
utils.py     # chat templating, model loading, thinking/answer token-range detection
extract.py   # Stage 1: rollouts → pooled activations → saved per-persona .pt files
analyze.py   # Stage 2: diff vectors, AUROC, plots, candidate selection, manifest
```

Expected output layout after both stages run:

```
extracted/
  naive_friend.pt          # pos/neg pooled activation stacks, [N, L, D] per strategy
  left_leaning.pt
  ...
analysis_out/
  naive_friend.png                       # AUROC + norm + cosine plots
  naive_friend_thinking_diff.pt          # [L, D] diff vector
  naive_friend_answer_diff.pt
  naive_friend_combined_diff.pt
  naive_friend_manifest.json             # candidate layers per strategy + AUROC peaks
  ...
```

## Implementation notes

- **`utils.find_thinking_answer_ranges`** — locates the response-relative ranges for the thinking and answer sections by scanning for Gemma-4 control tokens (`<|channel>`, `<channel|>`, `<turn|>`). Returns `None` for either section when it is missing or empty.
- **`extract.pool_hidden_states`** — mean-pools hidden states over each range. `combined` is pooled over the concatenation of both ranges (not the average of the two pooled vectors), so contributions stay length-weighted.
- **`analyze.compute_diff_vectors`** — per-layer `mean(pos) − mean(neg)` over the rollout dimension.
- **`analyze.compute_per_layer_auroc`** — per-layer projection of every rollout onto the diff vector, then `sklearn.metrics.roc_auc_score`. Asserts mean AUROC > 0.5 across layers as a directional sanity check.
- **`analyze.select_candidate_layers`** — 3-point moving-average smoothing of AUROC, then top-k descending. Layer 0 (embedding output) is masked when `skip_layer_zero` is set.
- **`analyze.plot_persona`** — three panels: AUROC vs layer (with peak annotations and candidate markers, plus a `y=0.5` chance line), diff-vector norm vs layer, and within-layer cosine across the three pooling-strategy pairs.

## Smoke test order

Before running on all five personas:
- Run `extract.py` on a single persona (edit `config.PERSONAS` to `["naive_friend"]`) with a tiny extraction range, e.g. temporarily set `EXTRACTION_QUESTION_RANGE = (0, 2)`. Verify the saved `.pt` has the expected shapes and no all-NaN strategies.
- Run `analyze.py` on that one persona. Verify AUROC values are above 0.5 (if they hover at 0.5, something is broken — usually the pos/neg are getting swapped or the pooling is wrong).
- Then expand to the full extraction split and the rest of the personas.

## What this scaffold deliberately does not do

- No multi-layer steering. Pick single-layer first; add multi-layer only if single-layer plateaus before reaching the persona expression you want.
- No α sweep — that goes in a stage 3 script. Mixing extraction (cheap, deterministic) with α sweeping (expensive, requires generation + judge calls) tangles iteration.
- No cross-layer cosine similarity. The basis of the residual stream changes layer to layer; cosine similarity across layers isn't meaningful. Only within-layer cosine (across pooling strategies) is computed.
- No judge-prompt evaluation. Plug that in at stage 3 alongside the α sweep.
