from transformers import AutoModelForImageTextToText
model_name = "llava-hf/vip-llava-7b-hf"
model = AutoModelForImageTextToText.from_pretrained(model_name)
with open(f"model_structure_{model_name.replace('/', '_')}.txt", "w") as f:
    f.write(str(model))