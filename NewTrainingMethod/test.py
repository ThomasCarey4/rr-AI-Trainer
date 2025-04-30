import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda")
input = torch.randint(0, 10000, (1, 128)).to("cuda")  # Fake input
print("Starting forward pass...")
output = model(input)  # Should complete instantly
print("Forward pass succeeded!")