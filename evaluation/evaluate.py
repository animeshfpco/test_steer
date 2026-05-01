"""
Held-out evaluation runner.

Given a `chosen_alpha.json` you write per-persona (after alpha tuning), runs:
    - unsteered baseline
    - mode "thinking_only" steered  (using its chosen alpha)
    - mode "thinking_answer" steered (using its chosen alpha)

over the held-out questions (default Q25–39). Outputs a JSONL with one record
per (persona, regime, mode, question) so you can read all conditions side by
side.

Expected `chosen_alpha.json` schema (you write this by hand after tuning):

  {
    "naive_friend": {
      "L14":    {"thinking_only": 1.5, "thinking_answer": 1.2},
      "L20_23": {"thinking_only": 0.8, "thinking_answer": 0.6}
    },
    "left_leaning": { ... },
    ...
  }

Any (persona, regime, mode) entry missing from this file is skipped.

Run:
    python -m evaluation.evaluate --chosen evaluation/chosen_alpha.json
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

import evaluation.configs as ec
from evaluation.steering import (
    PhaseGatedSteeringHook,
    gated_generate,
    load_diff_vectors_for_mode,
    GenerationTrace,
)
from src.steer_analysis.utils import (
    load_model_and_tokenizer,
    format_chat_with_system,
)
from src.steer_analysis.extract import load_persona


def banner(s: str) -> None:
    print("\n" + "=" * 80)
    print(s)
    print("=" * 80)


def trace_record(trace: GenerationTrace) -> dict:
    parsed = trace.parsed if isinstance(trace.parsed, dict) else {"raw": str(trace.parsed)}
    return {
        "text": trace.text,
        "parsed": {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                   for k, v in parsed.items()},
        "response_length": int(trace.response_token_ids.shape[0]),
        "final_phase": trace.final_phase,
        "phase_log": [asdict(t) for t in trace.phase_log],
    }


def evaluate_persona(
    model,
    tokenizer,
    persona: str,
    chosen: Dict[str, Dict[str, float]],
    eval_q_range,
    system_mode: str,
    max_new_tokens: int,
) -> None:
    data = load_persona(persona)
    questions = data["questions"]
    eval_qs = list(enumerate(questions))[eval_q_range[0] : eval_q_range[1]]

    if system_mode == "neutral":
        system_msg = "You are a helpful assistant."
    elif system_mode == "neg":
        system_msg = data["instruction"][0]["neg"]
    elif system_mode == "pos":
        system_msg = data["instruction"][0]["pos"]
    else:
        raise ValueError(f"Unknown system-mode {system_mode}")

    # Pre-load vectors for each requested (regime, mode) combo.
    vectors_cache: Dict[tuple, dict] = {}
    for regime_name, modes in chosen.items():
        if regime_name not in ec.REGIMES:
            print(f"  skip unknown regime {regime_name!r}")
            continue
        regime_layers = ec.REGIMES[regime_name].layers
        for mode in modes:
            if mode not in ec.MODES:
                print(f"  skip unknown mode {mode!r}")
                continue
            try:
                vectors_cache[(regime_name, mode)] = load_diff_vectors_for_mode(
                    persona, mode, regime_layers,
                )
            except FileNotFoundError as e:
                print(f"  missing diff vectors for {persona}/{mode}: {e}")

    # One JSONL per persona; one record per condition × question.
    out_root = ec.EVAL_DIR / persona
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "eval.jsonl"
    print(f"\n>>> writing {out_path}")
    with open(out_path, "w") as f_out:
        for q_idx, question in eval_qs:
            prompt_text = format_chat_with_system(
                tokenizer, system_msg=system_msg, user_msg=question,
                add_generation_prompt=True, enable_thinking=True,
            )

            # 1) Unsteered baseline (once per question).
            unsteered = gated_generate(
                model, tokenizer, prompt_text=prompt_text, hook=None,
                max_new_tokens=max_new_tokens, do_sample=ec.DO_SAMPLE,
            )
            f_out.write(json.dumps({
                "persona": persona, "regime": None, "mode": "unsteered",
                "alpha": 0.0, "question_index": q_idx, "question": question,
                "system_msg": system_msg, "model": ec.MODEL_NAME,
                "trace": trace_record(unsteered),
            }) + "\n")
            f_out.flush()

            # 2) Each (regime, mode) at its chosen alpha.
            for regime_name, modes in chosen.items():
                if regime_name not in ec.REGIMES:
                    continue
                regime_layers = ec.REGIMES[regime_name].layers
                for mode, alpha in modes.items():
                    key = (regime_name, mode)
                    if key not in vectors_cache:
                        continue
                    hook = PhaseGatedSteeringHook(
                        model=model, layers=regime_layers,
                        vectors=vectors_cache[key], alpha=float(alpha),
                    )
                    with hook:
                        steered = gated_generate(
                            model, tokenizer, prompt_text=prompt_text, hook=hook,
                            max_new_tokens=max_new_tokens, do_sample=ec.DO_SAMPLE,
                        )
                    f_out.write(json.dumps({
                        "persona": persona, "regime": regime_name, "mode": mode,
                        "alpha": float(alpha), "question_index": q_idx, "question": question,
                        "system_msg": system_msg, "model": ec.MODEL_NAME,
                        "trace": trace_record(steered),
                    }) + "\n")
                    f_out.flush()
                    print(f"  q{q_idx} {regime_name} {mode} α={alpha} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chosen", required=True, type=Path,
                        help="Path to chosen_alpha.json mapping persona->regime->mode->alpha.")
    parser.add_argument("--personas", nargs="*", default=None,
                        help="Subset of personas to run; default = all keys in chosen file.")
    parser.add_argument("--system-mode", choices=["neutral", "neg", "pos"], default="neutral")
    parser.add_argument("--max-new-tokens", type=int, default=ec.MAX_NEW_TOKENS)
    args = parser.parse_args()

    with open(args.chosen) as f:
        chosen_all: Dict[str, Dict[str, Dict[str, float]]] = json.load(f)

    target_personas = args.personas or list(chosen_all.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner(f"Loading {ec.MODEL_NAME} on {device}")
    model, tokenizer = load_model_and_tokenizer(ec.MODEL_NAME, device, ec.TORCH_DTYPE)

    for persona in target_personas:
        if persona not in chosen_all:
            print(f"skip {persona}: no entry in chosen file")
            continue
        banner(f"Evaluating {persona}")
        evaluate_persona(
            model=model, tokenizer=tokenizer, persona=persona,
            chosen=chosen_all[persona], eval_q_range=ec.EVAL_QUESTION_RANGE,
            system_mode=args.system_mode, max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
