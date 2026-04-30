from ast import Tuple

import pandas as pd
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma4ForConditionalGeneration
import numpy as np

MODEL_NAME = "google/gemma-4-E2B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
model.eval()
text_config = getattr(model.config, "text_config", model.config)
NUM_LAYERS = text_config.num_hidden_layers
D_MODEL = text_config.hidden_size


print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Num layers: {NUM_LAYERS}")
print(f"Hidden size: {D_MODEL}")

PAIRS = [
    ("How's the weather today?",
    "Arrr, the seas be calm and the winds be fair, matey!",
    "It's a pleasant day with mild temperatures and a gentle breeze."),

    ("What did you have for breakfast?",
    "Yarrr, I devoured a hearty plate o' salted hardtack and grog!",
    "I had some toast with jam and a cup of coffee."),

    ("Tell me about your job.",
    "Aye, I be sailin' the seven seas in search o' buried treasure!",
    "I work as a software engineer at a technology company."),

    ("What's your favorite hobby?",
    "Arrr, plunderin' merchant ships and singin' sea shanties be me joy!",
    "I enjoy reading books and going for walks in the park."),

    ("Where did you go on vacation?",
    "Yarrr, I sailed to Tortuga and drank rum 'til the dawn!",
    "I went to the Mediterranean and visited some historical sites."),
]

def format_chat(user_msg: str, assistant_msg: str) -> str:
    """Apply Qwen2.5 formatting to chat messages."""
    message = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    
    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

def get_response_hidden_states(user_msg: str, assistant_msg: str) -> torch.Tensor:
    """Return a tensor of hidden states for the response."""
    
    full_text = format_chat(user_msg, assistant_msg)
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(text=prompt_only, return_tensors="pt").to(DEVICE)
    full_ids = tokenizer(text=full_text, return_tensors="pt").to(DEVICE)
    
    response_start = prompt_ids["input_ids"].shape[-1]
    
    with torch.no_grad():
        outputs = model(
            **full_ids,
            output_hidden_states=True,
        )
    
    hidden = torch.stack(
        outputs.hidden_states[1:],
        dim=0,
    )
    
    response_hidden = hidden[:, 0, response_start:, :]
    meaned_pool = response_hidden.mean(1)
    
    return meaned_pool.cpu().float()

class SteeringHook:
    """
    Registers a forward hook on model.model.layers[layer_idx].
    During forward, adds alpha * v to that layer's output (residual stream).
    Apply at every token — generation is autoregressive, but the hook
    fires for whatever sequence length is being processed.
    """
    def __init__(self, model, layer_idx, v, alpha):
        self.model = model
        self.layer_idx = layer_idx
        self.v = v.to(DEVICE).to(model.dtype)
        self.alpha = alpha
        self.handle = None

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h_steered = h + self.alpha * self.v
            return (h_steered,) + output[1:]
        else:
            return output + self.alpha * self.v
    
    def __enter__(self):
        if isinstance(model, Gemma4ForConditionalGeneration):
            layer = self.model.model.language_model.layers[self.layer_idx]
        else:
            layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self
    
    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()


def generate(model, prompt, max_new_tokens=1000) -> tuple[str, dict]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=False)
    result = tokenizer.parse_response(response)
    return response, result

if __name__ == "__main__":
    diffs = []
    for users, pirate, normal in PAIRS:
        h_pirate = get_response_hidden_states(users, pirate)
        h_normal = get_response_hidden_states(users, normal)
        diff = h_pirate - h_normal
        diffs.append(diff)

    diffs = torch.stack(diffs, dim=0)
    v_trait_per_layer = diffs.mean(dim=0)

    norms = v_trait_per_layer.norm(dim=1)
    torch.save(v_trait_per_layer, "steer_vec/v_trait_per_layer.pt")
    print("Stored v_trait_per_layer.pt")

    LAYERS_TO_TRY = [10, 12]
    ALPHAS_TO_TRY = [1.75, 2.0, 2.25, 2.5]
    RESULTS = []

    print("Test promt: 'Good evening. hows been the day? write a 3 paraghraph essay on the day.'")
    print("─" * 80)
    for layer in LAYERS_TO_TRY:
        for alpha in ALPHAS_TO_TRY:
            with SteeringHook(model, layer, v_trait_per_layer[layer], alpha):
                out, res = generate(model,"Good evening. hows been the day? write a 3 paraghraph essay on the day.")
            # out = generate(model,"Good evening. hows been the day?", max_new_tokens=60)
            print(f"L={layer:2d} α={alpha:.1f} → ")
            for key, value in res.items():
                if key == "role":
                    print(f"Role: {value}")
                elif key == "thinking":
                    print(f"\n=== Thoughts ===\n{value}")
                elif key == "content":
                    print(f"\n=== Answer ===\n{value}")
                elif key == "tool_calls":
                    print(f"\n=== Tool Calls ===\n{value}")
                else:
                    print(f"\n{key}: {value}...\n")
            print("*" * 80)
            RESULTS.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "response": res,
                }
            )

    with open("results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    pd.DataFrame(RESULTS).to_csv("results.csv", index=False)