import json
from transformers import pipeline
from PIL import Image
import re

with open("mini_test_ordering.json", "r") as f:
    sample = json.load(f)
data = sample["data"][0]
answer_idx = data["answer"] 
print("Ground truth answer index (1-based):", answer_idx)
choice_groups = data["choice_list"]
correct_group = choice_groups[answer_idx - 1]  # 0-based
print("\nCorrect image file group:")
print(correct_group)

base_path = "./recipeqa/test/images-qa/"
img_paths = [base_path + name for name in correct_group]

print("\nFull image paths:")
print(img_paths)

# --- model ---
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
                    "Here are some images. Please sort them into the correct order. "
                    "Only output the filenames separated by spaces."
                )
            }
        ] + [
            {"type": "image", "image": Image.open(p)}
            for p in img_paths
        ]
    }
]

out = pipe(text=messages, max_new_tokens=100)
# print("\n=== Raw Model Output ===")
# print(out)

gen_list = out[0]["generated_text"]
assistant_msg = gen_list[-1]
text = assistant_msg["content"]

print("\n=== Assistant Text ===")
print(text)
text = text.strip()

pattern = r"how-to-cook-fava-beans_\d_\d\.jpg"
predicted_order = re.findall(pattern, text)

print("\nExtracted filenames from model output:")
print(predicted_order)

pred_indices = [
    correct_group.index(name) + 1
    for name in predicted_order
    if name in correct_group
]

print("\nPredicted ORDER indices (1~4):")
print(pred_indices)

expected = [1, 2, 3, 4]
print("\nExpected ORDER:", expected)

if pred_indices == expected:
    print("\n✅ CORRECT ANSWER")
else:
    print("\n❌ INCORRECT ANSWER")