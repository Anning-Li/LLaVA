import json
from transformers import pipeline
from PIL import Image
import re

with open("mini_test_coherence.json", "r") as f:
    sample = json.load(f)
data = sample["data"][0]
answer_idx = data["answer"]
print("Ground truth incoherent image index (1-based):", answer_idx)

choice_list = data["choice_list"]
print("\nChoice list:")
print(choice_list)
base_path = "./recipeqa/test/images-qa/"
img_paths = [base_path + name for name in choice_list]

print("\nFull image paths:")
print(img_paths)

pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-7b-hf",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Here are several images. One of them does NOT belong to this recipe. "
                    "Please identify the filename of the image that is inconsistent. "
                    "Only output the filename, nothing else."
                )
            }
        ] + [
            {"type": "image", "image": Image.open(p)}
            for p in img_paths
        ]
    }
]

out = pipe(text=messages, max_new_tokens=50)

gen_list = out[0]["generated_text"]
assistant_msg = gen_list[-1]
text = assistant_msg["content"].strip()

print("\n=== Model Output ===")
print(text)
pattern = r"[a-zA-Z0-9\-\_]+_\d_\d\.jpg"
predicted = re.findall(pattern, text)

print("\nExtracted filenames from model output:")
print(predicted)

pred_indices = [
    choice_list.index(name) + 1
    for name in predicted
    if name in choice_list
]

print("\nPredicted incoherent index:", pred_indices)

if pred_indices and pred_indices[0] == answer_idx:
    print("\n✅ CORRECT ANSWER")
else:
    print("\n❌ INCORRECT ANSWER")