from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the following images about?"},
            {"type": "image", "image": "./recipeqa/test/images-qa/apple-cake_1_0.jpg"},
            # {"type": "image", "image": "./recipeqa/test/images-qa/apple-cake_2_0.jpg"},
            # {"type": "image", "image": "./recipeqa/test/images-qa/apple-cake_3_1.jpg"},
            # {"type": "image", "image": "./recipeqa/test/images-qa/apple-cake_4_1.jpg"},
            # {"type": "image", "image": "./recipeqa/test/images-qa/apple-cake_5_0.jpg"},
        ]
    }
]

out = pipe(text=messages, max_new_tokens=50)

gen_list = out[0]["generated_text"]
assistant_msg = gen_list[-1]
text = assistant_msg["content"].strip()

print("\n=== Model Output ===")
print(text)