import json
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re

with open("mini_test_ordering.json", "r") as f:
    sample = json.load(f)

data = sample["data"][0]
answer_idx = data["answer"]
print("Ground truth answer index (0-based):", answer_idx)

choice_groups = data["choice_list"]
correct_group = choice_groups[answer_idx]
print("\nCorrect image filename group:")
print(correct_group)

base_path = "./recipeqa/test/images-qa/"
img_paths = [base_path + name for name in correct_group]

print("\nFull image paths:")
print(img_paths)

# ================================
# Load LLaVA model
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", use_fast=False
)
model = AutoModelForVision2Seq.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# ================================
# Step 1 — Explain Each Image
# ================================
pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-7b-hf",
    device=0,
)

descriptions = []
for i, p in enumerate(img_paths, start=1):
    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Describe this image briefly."},
            {"type": "image", "image": p},
        ]
    }]
    out = pipe(text=msg, max_new_tokens=50)
    text = out[0]["generated_text"][-1]["content"].strip()
    descriptions.append(text)

print("\n--- descriptions ---")
for i, cap in enumerate(descriptions):
    print(f"[{i}] {correct_group[i]}")
    print("Caption:", cap, "\n")

# ================================
# Step 2 — Ask model to sort based on descriptions
# ================================
print("\n=== Step 2: Ask Model to Sort ===")

desc_text_block = ""
for i, d in enumerate(descriptions):
    desc_text_block += f"{i}: {d}\n"

order_prompt = (
    "Here are descriptions of four images:\n"
    f"{desc_text_block}\n"
    "Sort the image indices (0-3) into the correct logical order. "
    "Only output space-separated numbers like: 0 1 2 3"
)

inputs = processor(text=order_prompt, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=30)
order_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("\n=== Model Sorting Output ===")
print(order_text)

# ================================
# Parse predicted order
# ================================
nums = re.findall(r"\d+", order_text)
pred_order = [int(n) for n in nums[:4]]

print("\nPredicted ORDER (0-based indices):", pred_order)

expected = [0, 1, 2, 3]  # because correct_group is the correct ordering itself
print("Expected ORDER:", expected)

if pred_order == expected:
    print("\n✅ CORRECT ORDER")
else:
    print("\n❌ INCORRECT ORDER")