"""
Phase-gated activation steering for Gemma 4.

Two pieces:

1. PhaseGatedSteeringHook
   Forward hooks on a configurable set of transformer layers. Each hook
   adds `alpha * v_phase[layer]` to the residual stream when steering is
   active for the current phase. Phase is set externally via
   `hook.set_phase(...)`. Phases:
       "thinking"  → uses thinking diff vectors (if provided)
       "answer"    → uses answer diff vectors (if provided)
       "off"       → no injection

2. gated_generate(...)
   Custom token-by-token generation loop using the model's KV cache.
   Detects Gemma 4 control tokens in the just-emitted token id and flips
   the hook's phase BEFORE the next forward pass. Replaces
   `model.generate(...)` for steering — `generate()` does not give us a
   per-step hook, so we drive sampling ourselves.

Why this works:
  - The chat template with enable_thinking=True pre-opens the thinking
    section, so generation starts with phase = "thinking".
  - The model emits `<channel|>` to switch to the answer; we observe that
    token and flip phase to "answer".
  - The model emits `<turn|>` to end; we stop and flip phase to "off".

If a `vectors` dict is missing a phase key (e.g. mode "thinking_only"
omits "answer"), that phase is treated as "off" automatically.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import Gemma4ForConditionalGeneration

from src.steer_analysis.utils import (
    GEMMA4_THINKING_OPEN,
    GEMMA4_THINKING_CLOSE,
    GEMMA4_TURN_CLOSE,
    _resolve_control_token_ids,
)


PhaseName = str  # "thinking" | "answer" | "off"


def _get_layer_module(model, layer_idx: int):
    """Resolve transformer block for either Gemma4 conditional-gen or vanilla causal LM."""
    if isinstance(model, Gemma4ForConditionalGeneration):
        return model.model.language_model.layers[layer_idx]
    return model.model.layers[layer_idx]


@dataclass
class PhaseGatedSteeringHook:
    """Multi-layer phase-gated steering hook.

    Args:
        model: HF model
        layers: list of transformer-block indices to hook (e.g. [14] or [20,21,22,23])
        vectors: phase → {layer_idx: tensor[D]}. Phases without an entry are no-ops.
                 e.g. {"thinking": {14: v_think_l14}, "answer": {14: v_ans_l14}}
        alpha: scalar coefficient applied at every hooked layer.
    """
    model: object
    layers: List[int]
    vectors: Dict[PhaseName, Dict[int, torch.Tensor]]
    alpha: float
    phase: PhaseName = "thinking"
    _handles: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Move all vectors to model device + dtype once.
        device = self.model.device
        dtype = self.model.dtype
        moved: Dict[PhaseName, Dict[int, torch.Tensor]] = {}
        for phase, layer_map in self.vectors.items():
            moved[phase] = {l: v.to(device).to(dtype) for l, v in layer_map.items()}
        self.vectors = moved

    def set_phase(self, phase: PhaseName) -> None:
        if phase not in ("thinking", "answer", "off"):
            raise ValueError(f"Unknown phase: {phase}")
        self.phase = phase

    def _make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            phase_vecs = self.vectors.get(self.phase)
            if phase_vecs is None:
                return output
            v = phase_vecs.get(layer_idx)
            if v is None:
                return output
            if isinstance(output, tuple):
                h = output[0]
                return (h + self.alpha * v,) + output[1:]
            return output + self.alpha * v
        return _hook

    def __enter__(self):
        for layer_idx in self.layers:
            block = _get_layer_module(self.model, layer_idx)
            handle = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Phase detector
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransition:
    """One observed phase change during generation. Token-relative to the response."""
    response_token_index: int
    triggering_token_id: int
    triggering_token_text: str
    new_phase: PhaseName


@dataclass
class GenerationTrace:
    """What gated_generate returns alongside the decoded text."""
    full_token_ids: torch.Tensor       # [seq_len], includes prompt
    prompt_length: int
    response_token_ids: torch.Tensor   # [response_len]
    text: str                           # decoded response (with special tokens)
    parsed: dict                        # tokenizer.parse_response output (thinking/content split)
    phase_log: List[PhaseTransition]   # in order
    final_phase: PhaseName


def _build_phase_machine(tokenizer):
    """Returns (initial_phase, transition_fn) where transition_fn(token_id) -> new_phase | None."""
    ids = _resolve_control_token_ids(tokenizer)
    open_id = ids[GEMMA4_THINKING_OPEN]
    close_id = ids[GEMMA4_THINKING_CLOSE]
    turn_id = ids[GEMMA4_TURN_CLOSE]

    def transition(current_phase: PhaseName, token_id: int) -> Optional[PhaseName]:
        # Reaching the close marker ends the thinking span. Inject ON the close
        # token itself can be argued either way; we choose to flip AFTER it has
        # been generated (i.e. for the next token, we are in "answer").
        if token_id == close_id:
            return "answer"
        if token_id == turn_id:
            return "off"
        # If the model re-opens thinking mid-response (unlikely but possible),
        # flip back. Pre-opened thinking from the chat template lives in the
        # prompt, so this branch is for in-response opens only.
        if token_id == open_id:
            return "thinking"
        return None

    return "thinking", transition, {"open": open_id, "close": close_id, "turn": turn_id}


def gated_generate(
    model,
    tokenizer,
    prompt_text: str,
    hook: Optional[PhaseGatedSteeringHook] = None,
    max_new_tokens: int = 800,
    do_sample: bool = False,
    temperature: float = 1.0,
    stop_on_turn_close: bool = True,
) -> GenerationTrace:
    """Token-by-token generation with phase-gated steering.

    If `hook` is None, runs an unsteered rollout (still records phase log so
    you can compare).
    """
    inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
    initial_phase, transition_fn, ctrl_ids = _build_phase_machine(tokenizer)
    turn_id = ctrl_ids["turn"]

    inputs = tokenizer(text=prompt_text, return_tensors="pt").to(model.device)
    input_ids: torch.Tensor = inputs["input_ids"]                # [1, L_p]
    attention_mask: torch.Tensor = inputs.get("attention_mask")
    prompt_length = input_ids.shape[-1]

    if hook is not None:
        hook.set_phase(initial_phase)

    phase = initial_phase
    phase_log: List[PhaseTransition] = []
    response_token_ids: List[int] = []

    past_key_values = None
    cur_input = input_ids
    cur_attn = attention_mask

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input,
                attention_mask=cur_attn,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)        # [1, 1]
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)             # [1, 1]

            tid = int(next_id.item())
            response_token_ids.append(tid)

            # Update phase BEFORE the next forward, based on what was just emitted.
            new_phase = transition_fn(phase, tid)
            if new_phase is not None and new_phase != phase:
                tok_text = inner_tok.convert_ids_to_tokens([tid])[0]
                phase_log.append(PhaseTransition(
                    response_token_index=step,
                    triggering_token_id=tid,
                    triggering_token_text=tok_text,
                    new_phase=new_phase,
                ))
                phase = new_phase
                if hook is not None:
                    hook.set_phase(phase)

            if stop_on_turn_close and tid == turn_id:
                break

            cur_input = next_id
            if cur_attn is not None:
                cur_attn = torch.cat(
                    [cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=cur_attn.device)],
                    dim=1,
                )

    response_tensor = torch.tensor(response_token_ids, dtype=input_ids.dtype, device=input_ids.device)
    full_ids = torch.cat([input_ids[0], response_tensor], dim=0)

    text = tokenizer.decode(response_tensor, skip_special_tokens=False)
    try:
        parsed = tokenizer.parse_response(text)
    except Exception:
        parsed = {"raw": text}

    return GenerationTrace(
        full_token_ids=full_ids,
        prompt_length=prompt_length,
        response_token_ids=response_tensor,
        text=text,
        parsed=parsed,
        phase_log=phase_log,
        final_phase=phase,
    )


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def load_diff_vectors_for_mode(
    persona: str,
    mode: str,
    regime_layers: List[int],
) -> Dict[PhaseName, Dict[int, torch.Tensor]]:
    """Load the diff-vector tensors needed for one (persona, mode, regime).

    `mode` resolves which pooling strategy to use per phase via configs.MODES.
    Returns a nested dict suitable for PhaseGatedSteeringHook.vectors.
    """
    from evaluation.configs import MODES, diff_vector_path

    if mode not in MODES:
        raise KeyError(f"Unknown mode {mode}. Known: {list(MODES)}")
    phase_to_strategy = MODES[mode]

    out: Dict[PhaseName, Dict[int, torch.Tensor]] = {}
    cache: Dict[str, torch.Tensor] = {}
    for phase, strategy in phase_to_strategy.items():
        if strategy == "off":
            continue
        if strategy not in cache:
            tensor = torch.load(diff_vector_path(persona, strategy), weights_only=False)  # [L, D]
            cache[strategy] = tensor
        diff = cache[strategy]
        out[phase] = {l: diff[l] for l in regime_layers}
    return out


@contextmanager
def maybe_steered(
    model,
    layers: List[int],
    vectors: Optional[Dict[PhaseName, Dict[int, torch.Tensor]]],
    alpha: float,
):
    """Context manager that yields a hook (or None for unsteered runs)."""
    if vectors is None or alpha == 0.0:
        yield None
        return
    hook = PhaseGatedSteeringHook(
        model=model, layers=layers, vectors=vectors, alpha=alpha,
    )
    with hook:
        yield hook
