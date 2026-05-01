"""
Evaluation configuration: layer regimes, injection modes, paths.

Two layer regimes:
  - L14         : single-layer injection at the raw AUROC peak.
  - L20_23      : band injection across layers 20–23 (alpha applied at each layer).

Two injection modes:
  - thinking_only : inject during the thinking span only; turn off at the
                    answer marker. Tests whether persona-shaped reasoning
                    survives once steering is removed.
  - thinking_answer : inject the thinking diff vector during the thinking
                      span and the answer diff vector during the answer span.
                      Tests joint reasoning + speech alignment.

Output layout:
  evaluation/results/<persona>/<regime>__<mode>/
      tuning_<alpha>.json      # one file per alpha tried during tuning
      eval.jsonl               # held-out evaluation, written by evaluate.py
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

import src.steer_analysis.config as base_config


REPO_ROOT = base_config.REPO_ROOT
ANALYSIS_DIR = base_config.ANALYSIS_DIR             # holds <persona>_<strategy>_diff.pt
EVAL_DIR = REPO_ROOT / "evaluation" / "results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ----- regimes -----
@dataclass(frozen=True)
class Regime:
    name: str
    layers: List[int]


REGIMES: Dict[str, Regime] = {
    "L14":     Regime(name="L14",    layers=[14]),
    "L20_23":  Regime(name="L20_23", layers=[20, 21, 22, 23]),
}


# ----- injection modes -----
# Each mode maps phase name -> pooling-strategy used for that phase.
# Phase "off" means no injection at that point.
MODES: Dict[str, Dict[str, str]] = {
    "thinking_only":   {"thinking": "thinking", "answer": "off"},
    "thinking_answer": {"thinking": "thinking", "answer": "answer"},
}


# ----- personas (mirror upstream config) -----
PERSONAS: List[str] = base_config.PERSONAS


# ----- generation -----
MAX_NEW_TOKENS = 800
DO_SAMPLE = False                # deterministic so alpha sweeps are comparable
TEMPERATURE = 1.0                # ignored when DO_SAMPLE is False
TORCH_DTYPE = base_config.TORCH_DTYPE
MODEL_NAME = base_config.MODEL_NAME


# ----- evaluation question split -----
EVAL_QUESTION_RANGE = base_config.EVAL_QUESTION_RANGE        # (25, 40)


# ----- helpers -----
def persona_result_dir(persona: str, regime: str, mode: str) -> Path:
    """results/<persona>/<regime>__<mode>/"""
    p = EVAL_DIR / persona / f"{regime}__{mode}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def diff_vector_path(persona: str, strategy: str) -> Path:
    """analysis_out/<persona>_<strategy>_diff.pt"""
    return ANALYSIS_DIR / f"{persona}_{strategy}_diff.pt"
