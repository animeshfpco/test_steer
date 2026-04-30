"""
Configuration for the steering vector extraction + analysis pipeline.

Edit values here rather than spreading them across scripts.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ----- paths -----
PROJECT_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = Path(__file__).parent.resolve()                          # src/steer_analysis/
REPO_ROOT = PROJECT_ROOT.parent.parent                                  # steer/
DATA_GEN_DIR = REPO_ROOT / "data_generation"
EXTRACTED_DIR = REPO_ROOT / "extracted"            # or keep under PROJECT_ROOT — your call
ANALYSIS_DIR = REPO_ROOT / "analysis_out"

EXTRACTED_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)


# ----- model -----
MODEL_NAME = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 1000        # generation budget per rollout — needs to be enough for thinking + answer
DO_SAMPLE = False            # deterministic for reproducibility during extraction
TORCH_DTYPE = "bfloat16"     # "bfloat16" on CUDA, "float32" on CPU


# ----- which personas to process -----
# Each entry must match a {persona}.json file in DATA_GEN_DIR.
PERSONAS = [
    "naive_friend",
    "left_leaning",
    "right_leaning",
    "gym_bro",
    "shy_nerd",
]


# ----- extraction split -----
# The persona JSONs have 40 questions. Use first half for extraction, second half for held-out eval.
EXTRACTION_QUESTION_RANGE = (0, 25)
EVAL_QUESTION_RANGE = (25, 40)


# ----- pooling strategies -----
# Three parallel tracks. Each rollout produces three pooled activation tensors.
POOLING_STRATEGIES = ["thinking", "answer", "combined"]


# ----- analysis -----
@dataclass
class AnalysisConfig:
    # Layers to analyze. None = all layers. Otherwise a list of layer indices.
    layers_to_analyze: List[int] | None = None
    # Number of candidate layers to recommend per pooling strategy after analysis.
    n_candidates: int = 3
    # Skip the embedding layer (layer 0 in stacked hidden_states is post-embedding).
    skip_layer_zero: bool = True

ANALYSIS = AnalysisConfig()
