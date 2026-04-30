"""
Shared utilities: chat templating, model loading, locating thinking vs answer
token ranges within a generated rollout.

Most of this is straightforward scaffolding. The non-trivial piece is
`find_thinking_answer_ranges` — it has to map Gemma-4's thinking tag tokens
to (start, end) index pairs in the full token sequence. That logic is
left as a TODO since it's model-specific and best verified interactively.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass
class TokenRanges:
    """Index ranges (start inclusive, end exclusive) into the full token sequence.

    All indices refer to positions in the [batch=0] dimension of input_ids/hidden_states
    AFTER the prompt — i.e. the response portion only. Caller is responsible for
    offsetting by prompt_length when slicing into the full forward-pass tensor.
    """
    prompt_length: int          # index where response starts in the full sequence
    response_length: int        # number of generated response tokens (excluding prompt)
    thinking: Optional[Tuple[int, int]]  # response-relative range, or None if no thinking
    answer: Optional[Tuple[int, int]]    # response-relative range, or None if no answer found


def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    """Load model + tokenizer matching the original script's setup."""
    tokenizer = AutoProcessor.from_pretrained(model_name, padding_side="left")
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def get_num_layers_and_hidden_size(model) -> Tuple[int, int]:
    """Return (num_layers, hidden_size) handling Gemma-4 nested config."""
    text_config = getattr(model.config, "text_config", model.config)
    return text_config.num_hidden_layers, text_config.hidden_size


def format_chat_with_system(
    tokenizer,
    system_msg: str,
    user_msg: str,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Build the full chat prompt string with a system message + user turn.

    Returns the formatted prompt up to and including the assistant turn marker
    (so generation continues from there). The persona pos/neg instruction
    goes in `system_msg`.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


# Gemma 4 control tokens (per the Gemma 4 prompt-formatting docs).
# A thinking-mode model turn looks like:
#     <|turn>model\n<|channel>thought\n ...thinking... <channel|>...answer...<turn|>
# When apply_chat_template is called with add_generation_prompt=True and
# enable_thinking=True, the template typically pre-opens the thinking section,
# so the model's first generated tokens are already inside the thinking content.
# We handle both cases (prompt pre-opens vs response opens) defensively below.
GEMMA4_THINKING_OPEN = "<|channel>"
GEMMA4_THINKING_CLOSE = "<channel|>"
GEMMA4_TURN_CLOSE = "<turn|>"


def _resolve_control_token_ids(tokenizer) -> dict:
    """Resolve Gemma 4 control tokens to single integer IDs. Errors loudly if any
    one fails to round-trip — that's a sign of a tokenizer or model mismatch and
    everything downstream depends on these being correct."""
    ids = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for name in (GEMMA4_THINKING_OPEN, GEMMA4_THINKING_CLOSE, GEMMA4_TURN_CLOSE):
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is None or (unk_id is not None and tid == unk_id):
            raise RuntimeError(
                f"Gemma 4 control token {name!r} did not resolve to a known "
                f"single-token id (got {tid}). Run `python utils.py` to inspect."
            )
        ids[name] = tid
    return ids


def find_thinking_answer_ranges(
    tokenizer,
    full_token_ids: torch.Tensor,   # [seq_len], the full sequence including prompt
    prompt_length: int,
) -> TokenRanges:
    """Locate the thinking-section and answer-section token ranges within the response.

    Implementation notes (Gemma 4):
      - Thinking-open `<|channel>` may or may not appear in the response: if the
        chat template pre-opened thinking via `enable_thinking=True`, generation
        starts inside the thinking section and the open token lives in the prompt.
      - Thinking-close `<channel|>` separates thinking from answer.
      - Turn-close `<turn|>` ends the answer (or the rollout cuts off without it
        if max_new_tokens was hit).

    Returns response-relative ranges. None for either field when:
      - thinking never closes (model ran out of budget mid-thought) → answer=None
      - response is empty                                            → both=None
      - either range computes as empty/inverted                      → that one=None
    """
    ids = _resolve_control_token_ids(tokenizer)
    open_id  = ids[GEMMA4_THINKING_OPEN]
    close_id = ids[GEMMA4_THINKING_CLOSE]
    turn_id  = ids[GEMMA4_TURN_CLOSE]

    response_ids = full_token_ids[prompt_length:].cpu()
    response_length = int(response_ids.shape[0])

    if response_length == 0:
        return TokenRanges(prompt_length, 0, thinking=None, answer=None)

    open_positions  = (response_ids == open_id).nonzero(as_tuple=True)[0]
    close_positions = (response_ids == close_id).nonzero(as_tuple=True)[0]
    turn_positions  = (response_ids == turn_id).nonzero(as_tuple=True)[0]

    # Thinking start: one past the first <|channel> in the response if present
    # (skip the open marker and the "thought\n" preamble lives in the next 1-2
    # tokens — pooling over them is harmless dilution on a long thinking trace).
    # If no open in response, the prompt opened thinking and content starts at 0.
    if len(open_positions) > 0:
        thinking_start = int(open_positions[0].item()) + 1
    else:
        thinking_start = 0

    # No close → thinking ran past max_new_tokens. Whole response is thinking,
    # answer is missing.
    if len(close_positions) == 0:
        thinking = (thinking_start, response_length) if response_length > thinking_start else None
        return TokenRanges(prompt_length, response_length, thinking=thinking, answer=None)

    thinking_end = int(close_positions[0].item())
    thinking = (thinking_start, thinking_end) if thinking_end > thinking_start else None

    # Answer: from one past <channel|> to the first <turn|> after it (or end).
    answer_start = thinking_end + 1
    later_turns = turn_positions[turn_positions > thinking_end]
    answer_end = int(later_turns[0].item()) if len(later_turns) > 0 else response_length
    answer = (answer_start, answer_end) if answer_end > answer_start else None

    return TokenRanges(prompt_length, response_length, thinking=thinking, answer=answer)


# ----- standalone verification -----
# Run `python utils.py` to confirm the tokenizer resolves the Gemma 4 control
# tokens to single token IDs and to inspect the token layout of a fake rollout.
# This is the cheap pre-flight check to run before kicking off extraction.
if __name__ == "__main__":
    import sys, config
    print(f"Loading tokenizer for {config.MODEL_NAME}...")
    tokenizer = AutoProcessor.from_pretrained(config.MODEL_NAME)

    print("\nResolving control tokens:")
    try:
        ids = _resolve_control_token_ids(tokenizer)
        for name, tid in ids.items():
            roundtrip = tokenizer.convert_ids_to_tokens([tid])
            print(f"  {name!r:>14} -> id={tid}  (round-trip: {roundtrip})")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("\nProbing chat template output (system + user + generation prompt, thinking enabled):")
    sample = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello."},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    print("---")
    print(sample)
    print("---")
    sample_ids = tokenizer(sample, return_tensors="pt")["input_ids"][0]
    print(f"prompt token count: {sample_ids.shape[0]}")
    print(f"prompt contains <|channel> (thinking pre-opened by template): "
          f"{(sample_ids == ids[GEMMA4_THINKING_OPEN]).any().item()}")