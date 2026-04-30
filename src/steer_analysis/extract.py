"""
Stage 1: rollout + pooled-activation extraction.

For each persona, for each instruction pair, for each extraction-split question:
    - Generate a rollout under the POS system prompt (and again under NEG).
    - Capture hidden states at every layer for every response-token position.
    - Pool those hidden states three ways: thinking-tokens-only, answer-tokens-only,
      both-combined. Each pool produces a [num_layers, hidden_size] tensor.
    - Save the pooled tensors. We do NOT save raw hidden states (too large).

Output: one .pt file per persona at EXTRACTED_DIR / f"{persona}.pt", containing
pooled activations stacked across rollouts plus a metadata dict.

Schema of the saved file:
    {
        "pos": {
            "thinking": Tensor [N, L, D],
            "answer":   Tensor [N, L, D],
            "combined": Tensor [N, L, D],
        },
        "neg": { ... same shape ... },
        "meta": {
            "persona": str,
            "model": str,
            "rollout_indices": List[(condition, pair_idx, question_idx)],
            "rollouts_with_missing_ranges": List[...],   # rollouts where thinking/answer not found
        },
    }
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import src.steer_analysis.config as config
from src.steer_analysis.utils import (
    load_model_and_tokenizer,
    get_num_layers_and_hidden_size,
    format_chat_with_system,
    find_thinking_answer_ranges,
    TokenRanges,
)


def load_persona(persona_name: str) -> dict:
    """Load a persona JSON from DATA_GEN_DIR."""
    path = config.DATA_GEN_DIR / "personas" / f"{persona_name}.json"
    with open(path) as f:
        return json.load(f)


def generate_rollout_with_hidden_states(
    model,
    tokenizer,
    system_msg: str,
    user_msg: str,
) -> Tuple[torch.Tensor, int]:
    """Run one rollout and return the full token sequence + prompt length.

    We do NOT capture hidden states during generate() because that gives us
    one set of hidden states per generated token (expensive and awkward to
    pool over). Instead: generate the response, then run ONE forward pass
    over the full sequence with output_hidden_states=True to get hidden
    states at every position cleanly. This is the same trick the original
    extraction script uses for its hand-written contrastive pairs.

    Returns:
        full_token_ids: [seq_len] tensor on model.device
        prompt_length: int — where the response starts in full_token_ids
    """
    prompt_text = format_chat_with_system(
        tokenizer, system_msg=system_msg, user_msg=user_msg,
        add_generation_prompt=True, enable_thinking=True,
    )
    prompt_ids = tokenizer(text=prompt_text, return_tensors="pt").to(model.device)
    prompt_length = prompt_ids["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **prompt_ids,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
        )
    # outputs shape: [1, prompt_length + response_length]
    full_token_ids = outputs[0]
    return full_token_ids, prompt_length


def hidden_states_for_full_sequence(
    model,
    full_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass over the full sequence and return hidden states.

    Returns:
        Tensor of shape [num_layers, seq_len, hidden_size] on CPU as float32.
        We exclude the embedding layer (index 0 in outputs.hidden_states) so
        layer 0 of the returned tensor corresponds to the output of transformer
        block 0.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=full_token_ids.unsqueeze(0).to(model.device),
            output_hidden_states=True,
        )
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors of shape [1, seq_len, hidden]
    # Index 0 = embedding output, indices 1..L = transformer block outputs.
    hidden = torch.stack(outputs.hidden_states[1:], dim=0)  # [L, 1, seq_len, D]
    hidden = hidden[:, 0, :, :]                              # [L, seq_len, D]
    return hidden.cpu().float()


def pool_hidden_states(
    hidden: torch.Tensor,         # [L, seq_len, D] over the full sequence
    prompt_length: int,
    ranges: TokenRanges,
) -> Dict[str, Optional[torch.Tensor]]:
    """Produce three pooled representations of one rollout, one per strategy.

    Each pooled tensor has shape [L, D] (one vector per layer).

    Returns a dict with keys 'thinking', 'answer', 'combined'. A value is None
    if its range is missing (e.g. thinking didn't close cleanly).

    TODO: implement the actual mean-pooling.
        For each strategy:
          - thinking: mean over hidden[:, prompt_length + thinking_start : prompt_length + thinking_end, :]
                      along the seq dimension → [L, D]
          - answer:   same idea using ranges.answer
          - combined: mean over the union of both ranges, NOT mean of the two pooled
                      vectors (that would weight them equally regardless of length).
                      Concretely: build a list of (start, end) spans, slice and concat
                      along seq dim, mean once over the result.
        If the corresponding range is None, return None for that key.
        Sanity check: pooled tensor has correct shape [L, D] and contains no NaNs.
    """
    out: Dict[str, Optional[torch.Tensor]] = {
        "thinking": None,
        "answer": None,
        "combined": None,
    }
    
    # thinking
    if ranges.thinking is not None:
        t_start, t_end = ranges.thinking
        thinking = hidden[:, prompt_length + t_start : prompt_length + t_end, :]
        out["thinking"] = thinking.mean(dim=1)

    # answer
    if ranges.answer is not None:
        a_start, a_end = ranges.answer
        answer = hidden[:, prompt_length + a_start : prompt_length + a_end, :]
        out["answer"] = answer.mean(dim=1)

    # combined
    if ranges.thinking is not None and ranges.answer is not None:
        t_start, t_end = ranges.thinking
        a_start, a_end = ranges.answer
        thinking = hidden[:, prompt_length + t_start : prompt_length + t_end, :]
        answer = hidden[:, prompt_length + a_start : prompt_length + a_end, :]
        combined = torch.cat([thinking, answer], dim=1)
        out["combined"] = combined.mean(dim=1)

    return out


def extract_for_persona(
    model,
    tokenizer,
    persona_name: str,
    num_layers: int,
    hidden_size: int,
) -> None:
    """Run extraction for one persona and save the result."""
    persona_data = load_persona(persona_name)
    instructions = persona_data["instruction"]      # list of {"pos":..., "neg":...} ×5
    questions = persona_data["questions"]           # list of 40 strings
    q_start, q_end = config.EXTRACTION_QUESTION_RANGE
    extraction_questions = questions[q_start:q_end]

    pooled_buffers: Dict[str, Dict[str, List[torch.Tensor]]] = {
        cond: {strat: [] for strat in config.POOLING_STRATEGIES}
        for cond in ("pos", "neg")
    }
    rollout_indices: List[Tuple[str, int, int]] = []
    skipped: List[Tuple[str, int, int, str]] = []

    for pair_idx, pair in enumerate(instructions):
        for cond in ("pos", "neg"):
            system_msg = pair[cond]
            for q_idx, question in enumerate(tqdm(
                extraction_questions,
                desc=f"{persona_name} pair{pair_idx} {cond}",
                leave=False,
            )):
                full_ids, prompt_len = generate_rollout_with_hidden_states(
                    model, tokenizer, system_msg=system_msg, user_msg=question,
                )
                ranges = find_thinking_answer_ranges(
                    tokenizer, full_ids, prompt_len,
                )

                hidden = hidden_states_for_full_sequence(model, full_ids)
                pooled = pool_hidden_states(hidden, prompt_len, ranges)

                # Record per-strategy. Skip strategies where pooling returned None.
                for strat in config.POOLING_STRATEGIES:
                    if pooled[strat] is None:
                        skipped.append((cond, pair_idx, q_idx, strat))
                        # Insert a NaN-filled placeholder so per-rollout indexing
                        # stays aligned across strategies. Analysis stage filters
                        # these out before computing means.
                        pooled_buffers[cond][strat].append(
                            torch.full((num_layers, hidden_size), float("nan"))
                        )
                    else:
                        pooled_buffers[cond][strat].append(pooled[strat])

                rollout_indices.append((cond, pair_idx, q_idx))

                # Free GPU memory aggressively — the forward pass over the full
                # sequence holds a lot of activations.
                del hidden
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Stack into [N, L, D] tensors per (condition, strategy).
    save_payload: dict = {"pos": {}, "neg": {}, "meta": {}}
    for cond in ("pos", "neg"):
        for strat in config.POOLING_STRATEGIES:
            stack = torch.stack(pooled_buffers[cond][strat], dim=0)  # [N, L, D]
            save_payload[cond][strat] = stack

    save_payload["meta"] = {
        "persona": persona_name,
        "model": config.MODEL_NAME,
        "rollout_indices": rollout_indices,
        "rollouts_with_missing_ranges": skipped,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }

    out_path = config.EXTRACTED_DIR / f"{persona_name}.pt"
    torch.save(save_payload, out_path)
    print(f"Saved {out_path} (pos N={len(pooled_buffers['pos']['thinking'])}, "
          f"neg N={len(pooled_buffers['neg']['thinking'])}, skipped={len(skipped)})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {config.MODEL_NAME} on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        config.MODEL_NAME, device, config.TORCH_DTYPE,
    )
    num_layers, hidden_size = get_num_layers_and_hidden_size(model)
    print(f"num_layers={num_layers} hidden_size={hidden_size}")

    for persona_name in config.PERSONAS:
        if (config.EXTRACTED_DIR / f"{persona_name}.pt").exists():
            print(f"Skipping {persona_name} (already extracted)")
            continue
        print(f"\n=== Extracting {persona_name} ===")
        extract_for_persona(
            model, tokenizer, persona_name, num_layers, hidden_size,
        )


if __name__ == "__main__":
    main()
