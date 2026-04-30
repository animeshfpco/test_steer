"""
Stage 2: per-layer analysis on extracted activations.

For each persona, for each pooling strategy:
  - Compute the per-layer diff vector: mean(pos pooled) - mean(neg pooled), per layer.
  - Compute the per-layer DISCRIMINABILITY: how well does projecting individual
    rollouts onto the diff vector separate pos from neg? Use AUROC.
    Higher AUROC at a layer = the trait is more cleanly represented at that layer.
  - Compute the per-layer diff-vector NORM (sanity check, not selection criterion).
  - Plot AUROC vs layer (three curves overlaid: thinking / answer / combined).
  - Plot diff-vector norm vs layer.
  - Plot cosine similarity between thinking-pooled and answer-pooled diff vectors
    at each layer (within-layer comparison only — never across layers).
  - Pick top-k candidate layers per strategy by AUROC.
  - Save the diff vectors and a manifest of recommended (layer, strategy) pairs.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

import src.steer_analysis.config as config


@dataclass
class LayerAnalysis:
    """Per-(persona, strategy) analysis output."""
    persona: str
    strategy: str
    diff_vectors: torch.Tensor       # [L, D]
    norms: np.ndarray                # [L]
    aurocs: np.ndarray               # [L]
    candidate_layers: List[int]      # top-k by AUROC


def load_extracted(persona_name: str) -> dict:
    path = config.EXTRACTED_DIR / f"{persona_name}.pt"
    return torch.load(path, weights_only=False)


def filter_valid_rollouts(stack: torch.Tensor) -> torch.Tensor:
    """Drop rollouts whose pooled tensor contains NaN (i.e. a missing range).

    Args:
        stack: [N, L, D]
    Returns:
        [N_valid, L, D]
    """
    # A rollout is valid if no element is NaN. Check across layers and dims.
    valid_mask = ~torch.isnan(stack).any(dim=(1, 2))
    return stack[valid_mask]


def compute_diff_vectors(
    pos_stack: torch.Tensor,        # [N_pos, L, D]
    neg_stack: torch.Tensor,        # [N_neg, L, D]
) -> torch.Tensor:
    """Mean over rollouts first, then subtract. Returns [L, D].

    Mean over rollouts (not over flattened tokens) so that long rollouts don't
    dominate. This is the canonical Anthropic-style diff vector for the persona.
    """
    valid_pos = filter_valid_rollouts(pos_stack)
    valid_neg = filter_valid_rollouts(neg_stack)
    
    if valid_pos.shape[0] == 0 or valid_neg.shape[0] == 0:
        raise ValueError("No valid rollouts.")
    
    pos_mean = valid_pos.mean(dim=0)
    neg_mean = valid_neg.mean(dim=0)
    diff = pos_mean - neg_mean
    
    assert diff.shape == (pos_stack.shape[1], pos_stack.shape[2])
    return diff


def compute_per_layer_norms(diff: torch.Tensor) -> np.ndarray:
    """L2 norm of the diff vector at each layer.

    Args:
        diff: [L, D]
    Returns:
        np.ndarray of shape [L]
    """
    return diff.norm(dim=-1).cpu().numpy()


def compute_per_layer_auroc(
    pos_stack: torch.Tensor,        # [N_pos, L, D] valid rollouts only
    neg_stack: torch.Tensor,        # [N_neg, L, D] valid rollouts only
    diff: torch.Tensor,             # [L, D]
) -> np.ndarray:
    """At each layer, project every rollout onto that layer's diff vector and
    compute AUROC for separating pos from neg by projection score.

    Args:
        pos_stack, neg_stack: per-rollout pooled activations
        diff: per-layer diff vector

    Returns:
        np.ndarray of shape [L] with AUROC in [0, 1] per layer. ~0.5 = no separation.
        Mean AUROC < 0.5 across layers would indicate the diff vector points the
        wrong way — asserted at the bottom as a sanity check.
    """
    from sklearn.metrics import roc_auc_score

    L = diff.shape[0]
    auroc = np.zeros(L)

    for l in range(L):
        denom = diff[l].norm().clamp(min=1e-8)
        pos_proj = (pos_stack[:, l, :] @ diff[l]) / denom
        neg_proj = (neg_stack[:, l, :] @ diff[l]) / denom

        scores = torch.cat([pos_proj, neg_proj]).numpy()
        labels = np.concatenate([
            np.ones(pos_proj.shape[0]),
            np.zeros(neg_proj.shape[0]),
        ])
        auroc[l] = roc_auc_score(labels, scores)

    assert auroc.mean() > 0.5, (
        f"mean AUROC {auroc.mean():.3f} <= 0.5 — diff vector likely points the wrong way"
    )
    return auroc


def compute_within_layer_cosine(
    diff_a: torch.Tensor,    # [L, D]
    diff_b: torch.Tensor,    # [L, D]
) -> np.ndarray:
    """Cosine similarity between two diff vectors at each layer (same layer only).

    Args:
        diff_a, diff_b: per-layer diff vectors from two pooling strategies

    Returns:
        np.ndarray of shape [L]. Cross-layer comparisons are not meaningful and
        not computed — the basis of the residual stream changes across layers.
    """
    a = diff_a / diff_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = diff_b / diff_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (a * b).sum(dim=-1).cpu().numpy()


def select_candidate_layers(aurocs: np.ndarray, k: int, skip_layer_zero: bool) -> List[int]:
    """Top-k layers by AUROC. Layer 0 is the embedding output by our convention
    in extract.py — usually skip it. AUROC is smoothed with a 3-point moving
    average before ranking to avoid picking spiky neighbors of a flat plateau.
    """
    # 3-point moving average to dampen single-layer spikes; edges padded by replication.
    padded = np.concatenate([aurocs[:1], aurocs, aurocs[-1:]])
    smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0

    candidate_pool = smoothed.copy()
    if skip_layer_zero:
        candidate_pool[0] = -np.inf

    # Top-k indices, ordered by AUROC descending.
    top_k = np.argsort(candidate_pool)[::-1][:k]
    return [int(i) for i in top_k]


def analyze_persona(persona_name: str) -> Dict[str, LayerAnalysis]:
    """Run the full analysis for one persona across all pooling strategies.

    Returns a dict keyed by strategy name → LayerAnalysis.
    """
    print(f"\n=== Analyzing {persona_name} ===")
    payload = load_extracted(persona_name)
    results: Dict[str, LayerAnalysis] = {}

    for strategy in config.POOLING_STRATEGIES:
        pos_stack = filter_valid_rollouts(payload["pos"][strategy])  # [N_pos, L, D]
        neg_stack = filter_valid_rollouts(payload["neg"][strategy])  # [N_neg, L, D]
        print(f"  {strategy}: pos N={pos_stack.shape[0]} neg N={neg_stack.shape[0]}")

        if pos_stack.shape[0] == 0 or neg_stack.shape[0] == 0:
            print(f"    skipping {strategy}: no valid rollouts")
            continue

        diff = compute_diff_vectors(pos_stack, neg_stack)             # [L, D]
        norms = compute_per_layer_norms(diff)                         # [L]
        aurocs = compute_per_layer_auroc(pos_stack, neg_stack, diff)  # [L]
        candidates = select_candidate_layers(
            aurocs, k=config.ANALYSIS.n_candidates,
            skip_layer_zero=config.ANALYSIS.skip_layer_zero,
        )

        results[strategy] = LayerAnalysis(
            persona=persona_name,
            strategy=strategy,
            diff_vectors=diff,
            norms=norms,
            aurocs=aurocs,
            candidate_layers=candidates,
        )
        print(f"    AUROC max={aurocs.max():.3f} at layer {int(aurocs.argmax())}; "
              f"candidates: {candidates}")

    return results


def plot_persona(
    persona_name: str,
    results: Dict[str, LayerAnalysis],
    out_dir: Path,
) -> None:
    """Three-panel plot per persona: AUROC vs layer, norm vs layer, and within-layer
    cosine across pooling-strategy pairs. Saved as {out_dir}/{persona_name}.png.
    """
    if not results:
        print(f"  no results to plot for {persona_name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"thinking": "tab:blue", "answer": "tab:orange", "combined": "tab:green"}

    # Panel 1: AUROC vs layer
    ax = axes[0]
    for strategy, analysis in results.items():
        layers = np.arange(analysis.aurocs.shape[0])
        color = colors.get(strategy, None)
        ax.plot(layers, analysis.aurocs, label=strategy, color=color)
        peak = int(np.argmax(analysis.aurocs))
        ax.scatter([peak], [analysis.aurocs[peak]], color=color, zorder=5)
        ax.annotate(
            f"L{peak}={analysis.aurocs[peak]:.2f}",
            xy=(peak, analysis.aurocs[peak]),
            xytext=(4, 4), textcoords="offset points", fontsize=8, color=color,
        )
        for cand in analysis.candidate_layers:
            ax.axvline(cand, color=color, alpha=0.15, linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{persona_name}: per-layer AUROC")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: diff-vector norm vs layer
    ax = axes[1]
    for strategy, analysis in results.items():
        layers = np.arange(analysis.norms.shape[0])
        ax.plot(layers, analysis.norms, label=strategy, color=colors.get(strategy))
    ax.set_xlabel("layer")
    ax.set_ylabel("‖diff‖")
    ax.set_title("diff-vector norm")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: within-layer cosine across pooling-strategy pairs
    ax = axes[2]
    pairs = [
        ("thinking", "answer"),
        ("thinking", "combined"),
        ("answer", "combined"),
    ]
    for a, b in pairs:
        if a in results and b in results:
            cos = compute_within_layer_cosine(
                results[a].diff_vectors, results[b].diff_vectors,
            )
            ax.plot(np.arange(cos.shape[0]), cos, label=f"cos({a}, {b})")
    ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.set_title("within-layer cosine across strategies")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{persona_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_diff_vectors_and_manifest(
    persona_name: str,
    results: Dict[str, LayerAnalysis],
    out_dir: Path,
) -> None:
    """Save per-strategy diff vectors and a JSON manifest of recommended layers.

    Diff vectors saved as: {out_dir}/{persona_name}_{strategy}_diff.pt
        Tensor shape [L, D].

    Manifest saved as: {out_dir}/{persona_name}_manifest.json
        {
          "persona": str,
          "candidates": {
              "thinking": [layer_idx, ...],
              "answer":   [layer_idx, ...],
              "combined": [layer_idx, ...],
          },
          "auroc_peaks": {
              "thinking": {"layer": int, "auroc": float},
              ...
          },
        }
    """
    manifest = {
        "persona": persona_name,
        "candidates": {},
        "auroc_peaks": {},
    }
    for strategy, analysis in results.items():
        torch.save(
            analysis.diff_vectors,
            out_dir / f"{persona_name}_{strategy}_diff.pt",
        )
        manifest["candidates"][strategy] = analysis.candidate_layers
        peak_layer = int(np.argmax(analysis.aurocs))
        manifest["auroc_peaks"][strategy] = {
            "layer": peak_layer,
            "auroc": float(analysis.aurocs[peak_layer]),
        }

    with open(out_dir / f"{persona_name}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    for persona_name in config.PERSONAS:
        if not (config.EXTRACTED_DIR / f"{persona_name}.pt").exists():
            print(f"Skipping {persona_name}: no extracted activations found.")
            continue
        results = analyze_persona(persona_name)
        plot_persona(persona_name, results, config.ANALYSIS_DIR)
        save_diff_vectors_and_manifest(persona_name, results, config.ANALYSIS_DIR)


if __name__ == "__main__":
    main()
