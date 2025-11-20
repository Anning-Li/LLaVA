import json
from transformers import pipeline
from PIL import Image
import torch

with open("mini_test_coherence.json", "r") as f:
    sample = json.load(f)
data = sample["data"][0]
answer_idx = data["answer"]
print("Ground truth:", answer_idx)
base_path = "./recipeqa/test/images-qa/"
img_paths = [base_path + name for name in data["choice_list"]]
print("\nFull image paths:")
print(img_paths)

# ---------- Load model ----------
pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-7b-hf",
    device=0,
)

# ---- STEP 1: caption each image ----
captions = []
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
    captions.append(text)

# ---- Print captions ----
print("\n--- Captions ---")
for i, cap in enumerate(captions):
    print(f"[{i}] {data['choice_list'][i]}")
    print("Caption:", cap, "\n")

# ---- STEP 2: Ask model which caption is inconsistent ----
combined_desc = "\n".join([f"{i}. {captions[i]}" for i in range(len(captions))])

judge_prompt = f"""
Here are descriptions of several images from a cooking recipe:

{combined_desc}

One description does NOT belong to the recipe because it is unrelated or inconsistent.
Which number is the inconsistent one? Only output the number.
"""

judge_msg = [{
    "role": "user",
    "content": [{"type": "text", "text": judge_prompt}]
}]


judge_out = pipe(text=judge_msg, max_new_tokens=20)
judge_text = judge_out[0]["generated_text"][-1]["content"].strip()
# print("\n=== Model Decision Output ===")
# print(judge_text)

import re
nums = re.findall(r"\d+", judge_text)
if nums:
    pred_idx = int(nums[0]) - 1 
else:
    pred_idx = -1

print("\nPredicted incoherent index:", pred_idx)
print("Expected:", answer_idx)

if pred_idx == answer_idx:
    print("\n✅ CORRECT")
else:
    print("\n❌ INCORRECT")