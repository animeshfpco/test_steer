"""
Interactive alpha tuner.

Workflow per session:
  1. Pick a persona from the list.
  2. Pick a regime (L14 or L20_23).
  3. Pick a mode (thinking_only or thinking_answer).
  4. Pick a question (one of the held-out eval questions, or a free-form prompt).
  5. Loop: enter alpha values one or many at a time. For each alpha:
       - Run an unsteered baseline ONCE per question (cached).
       - Run a steered rollout at that alpha.
       - Print thinking + answer for both side-by-side.
       - Save the result to:
            evaluation/results/<persona>/<regime>__<mode>/tuning_alpha=<a>_q<idx>.json
  6. Continue until the user types `done`. Then optionally pick a different
     question / persona / regime / mode and continue.

The script never auto-decides "best alpha" — that's your call. It just runs
the rollouts you ask for and records them so you can re-read later.

Run:
    python -m evaluation.tune_alpha
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def hr(s: str = "") -> None:
    print("-" * 80 + (f" {s}" if s else ""))


def prompt_choice(label: str, options: List[str], default: Optional[str] = None) -> str:
    while True:
        print(f"\n{label}:")
        for i, opt in enumerate(options):
            mark = " (default)" if opt == default else ""
            print(f"  [{i}] {opt}{mark}")
        raw = input(f"choose 0..{len(options)-1} or name (blank=default): ").strip()
        if raw == "" and default is not None:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(options):
                return options[idx]
        if raw in options:
            return raw
        print(f"  invalid; try again")


def prompt_alphas(label: str = "alphas to try") -> List[float]:
    raw = input(f"\n{label} (comma/space separated, or 'done'): ").strip()
    if raw.lower() in ("done", "quit", "exit", "q"):
        return []
    parts = [p for chunk in raw.split(",") for p in chunk.split()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            print(f"  skipping non-numeric: {p!r}")
    return out


def build_question_pool(persona_name: str, default_system_idx: int = 0) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[dict]]:
    """Returns (extraction_questions_indexed, eval_questions_indexed, instructions).

    Indices are absolute (i.e. extraction is 0..24, eval is 25..39 with the
    upstream config defaults). Easier to refer to questions by absolute index
    in saved files.
    """
    data = load_persona(persona_name)
    questions = data["questions"]
    extraction = list(enumerate(questions))[: ec.EVAL_QUESTION_RANGE[0]]
    eval_pool = list(enumerate(questions))[ec.EVAL_QUESTION_RANGE[0] : ec.EVAL_QUESTION_RANGE[1]]
    return extraction, eval_pool, data["instruction"]


def render_trace(label: str, trace: GenerationTrace, max_chars: int = 1500) -> None:
    hr(label)
    parsed = trace.parsed if isinstance(trace.parsed, dict) else {"raw": str(trace.parsed)}
    thinking = parsed.get("thinking") or parsed.get("reasoning") or ""
    content = parsed.get("content") or parsed.get("answer") or ""
    if thinking:
        print("THINKING:")
        print(str(thinking)[:max_chars])
        print()
    if content:
        print("ANSWER:")
        print(str(content)[:max_chars])
    if not thinking and not content:
        print("(parsed split unavailable; raw text follows)")
        print(trace.text[:max_chars])
    print()
    print(f"[final_phase={trace.final_phase}  response_tokens={trace.response_token_ids.shape[0]}  "
          f"transitions={len(trace.phase_log)}]")


def trace_to_record(trace: GenerationTrace) -> dict:
    parsed = trace.parsed if isinstance(trace.parsed, dict) else {"raw": str(trace.parsed)}
    return {
        "text": trace.text,
        "parsed": {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                   for k, v in parsed.items()},
        "response_length": int(trace.response_token_ids.shape[0]),
        "final_phase": trace.final_phase,
        "phase_log": [asdict(t) for t in trace.phase_log],
    }


def save_result(
    persona: str, regime: str, mode: str, q_index: int, alpha: float,
    question: str, system_msg: str,
    unsteered: GenerationTrace, steered: GenerationTrace,
) -> Path:
    out_dir = ec.persona_result_dir(persona, regime, mode)
    fname = f"tuning_q{q_index:02d}_alpha={alpha:g}.json"
    path = out_dir / fname
    payload = {
        "persona": persona,
        "regime": regime,
        "mode": mode,
        "question_index": q_index,
        "question": question,
        "system_msg_used": system_msg,
        "alpha": alpha,
        "model": ec.MODEL_NAME,
        "max_new_tokens": ec.MAX_NEW_TOKENS,
        "do_sample": ec.DO_SAMPLE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unsteered": trace_to_record(unsteered),
        "steered":   trace_to_record(steered),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default=None,
                        help=f"Optional. One of {ec.PERSONAS}.")
    parser.add_argument("--regime", default=None, choices=list(ec.REGIMES.keys()))
    parser.add_argument("--mode", default=None, choices=list(ec.MODES.keys()))
    parser.add_argument("--system-mode", choices=["neutral", "neg"], default="neutral",
                        help="System prompt for the rollout. 'neutral' = generic helpful "
                             "assistant; 'neg' = the persona's NEG instruction (so steering "
                             "has to overcome an opposing prompt — harder test).")
    parser.add_argument("--max-new-tokens", type=int, default=ec.MAX_NEW_TOKENS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner(f"Loading {ec.MODEL_NAME} on {device}")
    model, tokenizer = load_model_and_tokenizer(ec.MODEL_NAME, device, ec.TORCH_DTYPE)

    # Outer loop: persona/regime/mode/question selection. Inner loop: alpha sweep.
    while True:
        persona = args.persona or prompt_choice("Persona", ec.PERSONAS, default=ec.PERSONAS[0])
        regime = args.regime or prompt_choice("Layer regime", list(ec.REGIMES.keys()), default="L14")
        mode = args.mode or prompt_choice("Injection mode", list(ec.MODES.keys()), default="thinking_only")
        regime_layers = ec.REGIMES[regime].layers

        # Load diff vectors for this (persona, mode, regime) once.
        try:
            vectors = load_diff_vectors_for_mode(persona, mode, regime_layers)
        except FileNotFoundError as e:
            print(f"  missing diff vectors: {e}")
            print("  run analyze.py first; aborting.")
            sys.exit(1)
        print(f"\nLoaded vectors. phases={list(vectors.keys())}  layers={regime_layers}")

        extraction, eval_pool, instructions = build_question_pool(persona)
        # Default system message: neutral. If --system-mode neg, use NEG of pair 0.
        system_msg = (
            "You are a helpful assistant."
            if args.system_mode == "neutral"
            else instructions[0]["neg"]
        )
        print(f"system_msg = {system_msg!r}")

        # Question loop.
        while True:
            print("\nHeld-out eval questions:")
            for q_idx, q in eval_pool[:10]:
                print(f"  [{q_idx}] {q[:90]}")
            if len(eval_pool) > 10:
                print(f"  ... ({len(eval_pool) - 10} more)")
            raw = input("\nQuestion index, or type freeform after '>', or 'next' to switch persona/regime/mode, or 'quit': ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                print("bye.")
                return
            if raw.lower() in ("next", "n"):
                # Reset CLI overrides so user can pick fresh.
                args.persona = None
                args.regime = None
                args.mode = None
                break
            if raw.startswith(">"):
                question = raw[1:].strip()
                q_index = -1
            elif raw.isdigit():
                q_index = int(raw)
                # Allow either eval index (25..39) or extraction (0..24) — we just look it up.
                all_qs = dict(extraction + eval_pool)
                if q_index not in all_qs:
                    print(f"  unknown index {q_index}")
                    continue
                question = all_qs[q_index]
            else:
                print("  could not parse; try again")
                continue

            prompt_text = format_chat_with_system(
                tokenizer, system_msg=system_msg, user_msg=question,
                add_generation_prompt=True, enable_thinking=True,
            )

            # Cache unsteered for this question.
            print("\n[unsteered baseline]")
            unsteered = gated_generate(
                model, tokenizer, prompt_text=prompt_text, hook=None,
                max_new_tokens=args.max_new_tokens, do_sample=ec.DO_SAMPLE,
            )
            render_trace("UNSTEERED", unsteered)

            # Alpha loop.
            while True:
                alphas = prompt_alphas("alpha values (or 'done' to pick a new question)")
                if not alphas:
                    break
                for alpha in alphas:
                    banner(f"{persona}  {regime}  {mode}  q{q_index}  alpha={alpha}")
                    hook = PhaseGatedSteeringHook(
                        model=model, layers=regime_layers, vectors=vectors, alpha=alpha,
                    )
                    with hook:
                        steered = gated_generate(
                            model, tokenizer, prompt_text=prompt_text, hook=hook,
                            max_new_tokens=args.max_new_tokens, do_sample=ec.DO_SAMPLE,
                        )
                    render_trace(f"STEERED alpha={alpha}", steered)
                    saved = save_result(
                        persona=persona, regime=regime, mode=mode, q_index=q_index,
                        alpha=alpha, question=question, system_msg=system_msg,
                        unsteered=unsteered, steered=steered,
                    )
                    print(f"saved -> {saved.relative_to(ec.REPO_ROOT)}")


if __name__ == "__main__":
    main()
