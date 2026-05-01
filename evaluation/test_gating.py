"""
Standalone smoke test for PhaseGatedSteeringHook + gated_generate.

Runs ONE rollout (unsteered) and prints:
  - the chat prompt
  - the full token-by-token decoded sequence
  - which token triggered each phase transition
  - the final phase and parsed thinking/answer split (per tokenizer.parse_response)

Then runs ONE steered rollout with a tiny alpha to verify the hook actually
fires per phase. We don't validate behavioral change here — only that the
state machine transitions on the expected control tokens.

Run with:
    python -m evaluation.test_gating

It picks the first persona in PERSONAS by default. Override with --persona.
"""
from __future__ import annotations
import argparse
import sys

import torch

import evaluation.configs as ec
from evaluation.steering import (
    PhaseGatedSteeringHook,
    gated_generate,
    load_diff_vectors_for_mode,
)
from src.steer_analysis.utils import (
    load_model_and_tokenizer,
    format_chat_with_system,
    _resolve_control_token_ids,
    GEMMA4_THINKING_OPEN,
    GEMMA4_THINKING_CLOSE,
    GEMMA4_TURN_CLOSE,
)


SAMPLE_QUESTION = "What's your view on the latest tech news from this week?"
SAMPLE_SYSTEM = "You are a helpful assistant."


def banner(s: str) -> None:
    print("\n" + "=" * 80)
    print(s)
    print("=" * 80)


def show_response_with_phase_log(tokenizer, trace, label: str) -> None:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    print(f"\n--- {label} ---")
    print(f"prompt_length = {trace.prompt_length}")
    print(f"response_length = {trace.response_token_ids.shape[0]}")
    print(f"final_phase = {trace.final_phase}")
    print(f"\nphase transitions ({len(trace.phase_log)}):")
    for t in trace.phase_log:
        print(
            f"  step={t.response_token_index:4d}  "
            f"tok_id={t.triggering_token_id:>6d}  "
            f"tok={t.triggering_token_text!r:>20}  "
            f"-> {t.new_phase}"
        )
    print("\n--- decoded response (first 600 chars) ---")
    print(trace.text[:600])
    print("\n--- parsed ---")
    if isinstance(trace.parsed, dict):
        for k, v in trace.parsed.items():
            preview = str(v)[:200]
            print(f"  {k}: {preview}")


def verify_transitions(trace, ctrl_ids: dict) -> None:
    """Sanity assertions: every transition should be on a known control token."""
    known = {
        ctrl_ids[GEMMA4_THINKING_OPEN]: "thinking-open",
        ctrl_ids[GEMMA4_THINKING_CLOSE]: "thinking-close",
        ctrl_ids[GEMMA4_TURN_CLOSE]: "turn-close",
    }
    for t in trace.phase_log:
        assert t.triggering_token_id in known, (
            f"Phase transition fired on non-control token id={t.triggering_token_id}"
        )
    print("\nOK: all phase transitions fired on Gemma 4 control tokens.")

    # If thinking ever closed, we expect at least one transition into "answer".
    seen_phases = [t.new_phase for t in trace.phase_log]
    if "answer" in seen_phases:
        print("OK: saw transition into 'answer' phase.")
    else:
        print(
            "WARN: no transition into 'answer' phase observed. "
            "Either thinking ran past max_new_tokens or the close marker was not emitted."
        )


def run_steered_smoke_test(
    model,
    tokenizer,
    prompt_text: str,
    persona: str,
    regime_name: str,
    mode: str,
    alpha: float,
) -> None:
    """One steered run with low alpha to confirm the hook context-manages cleanly
    and that phase flipping actually drives the hook (no exception, no crash)."""
    regime = ec.REGIMES[regime_name]
    vectors = load_diff_vectors_for_mode(persona, mode, regime.layers)
    print(f"\nLoaded vectors. phases={list(vectors.keys())} layers={list(next(iter(vectors.values())).keys())}")

    hook = PhaseGatedSteeringHook(
        model=model, layers=regime.layers, vectors=vectors, alpha=alpha,
    )
    with hook:
        trace = gated_generate(
            model, tokenizer, prompt_text=prompt_text, hook=hook,
            max_new_tokens=ec.MAX_NEW_TOKENS, do_sample=False,
        )
    show_response_with_phase_log(tokenizer, trace, f"steered ({persona} {regime_name} {mode} alpha={alpha})")
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default=ec.PERSONAS[0],
                        help=f"Persona to load diff vectors from. One of {ec.PERSONAS}.")
    parser.add_argument("--regime", default="L14", choices=list(ec.REGIMES.keys()))
    parser.add_argument("--mode", default="thinking_only", choices=list(ec.MODES.keys()))
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Small alpha just to exercise the hook.")
    parser.add_argument("--question", default=SAMPLE_QUESTION)
    parser.add_argument("--system", default=SAMPLE_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    banner(f"Loading {ec.MODEL_NAME} on {device}")
    model, tokenizer = load_model_and_tokenizer(ec.MODEL_NAME, device, ec.TORCH_DTYPE)
    ctrl_ids = _resolve_control_token_ids(tokenizer)
    print(f"control token ids: {ctrl_ids}")

    prompt_text = format_chat_with_system(
        tokenizer, system_msg=args.system, user_msg=args.question,
        add_generation_prompt=True, enable_thinking=True,
    )
    banner("Prompt (decoded chat template)")
    print(prompt_text)

    banner("Unsteered rollout (hook=None)")
    unsteered = gated_generate(
        model, tokenizer, prompt_text=prompt_text, hook=None,
        max_new_tokens=args.max_new_tokens, do_sample=False,
    )
    show_response_with_phase_log(tokenizer, unsteered, "unsteered")
    verify_transitions(unsteered, ctrl_ids)

    banner("Steered smoke test")
    try:
        run_steered_smoke_test(
            model, tokenizer, prompt_text=prompt_text,
            persona=args.persona, regime_name=args.regime,
            mode=args.mode, alpha=args.alpha,
        )
    except FileNotFoundError as e:
        print(
            f"\nSKIPPED steered run — missing diff vectors: {e}\n"
            f"Run analyze.py first (already done if analysis_out/ has *_diff.pt)."
        )
        sys.exit(0)
    print("\nAll gating checks passed.")


if __name__ == "__main__":
    main()
