import torch
from transformers import AutoModelForImageTextToText

model_name = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
model = AutoModelForImageTextToText.from_pretrained(model_name,dtype=torch.bfloat16)
print(model)
print("\n\n====================\n\n")
print(model.config)
print("\n\n====================\n\n")