import torch
from scripts.steer_and_test import SteeringHook, generate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STEER_VEC_PATH = "steer_vec/v_trait_per_layer.pt"
v_trait = torch.load(STEER_VEC_PATH)

LAYER_IDX = 13
ALPHA = 1.6

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
model.eval()

TEST_PROMPTS = [
    # "Good evening. hows been the day?",
    # "You are a gentleman. Good evening. hows been the day?",
    # "You are a artist. Good evening. hows been the day?",
    # "You are a software engineer. Good evening. hows been the day?",
    # "You are luffy from one piece. Good evening. hows been the day?",
    # "You are 17th century english pirate. Good evening. hows been the day?",
    "Write a 3 paragraph story about a kid named harry.",
]

print("─" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Layer: {LAYER_IDX}")
print(f"Alpha: {ALPHA}")
print("─" * 80)

for prompt in TEST_PROMPTS:
    with SteeringHook(model, LAYER_IDX, v_trait[LAYER_IDX], ALPHA):
        out = generate(model, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {out}")
    print("─" * 80)
