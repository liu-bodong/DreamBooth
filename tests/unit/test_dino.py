import torch
import torchvision
from torchvision import transforms
import PIL.Image as Image
import pathlib
import numpy as np

from src.evaluation.DINO import DINOEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# x = transforms.ToTensor()(x).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

evaluator = DINOEvaluator(device)

embeds = evaluator.embed([x])
print("DINO embedding shape:", embeds.shape)
assert embeds.shape == (1, 384), "Expected embedding shape to be (1, 384)"

assert torch.allclose(embeds.norm(dim=1), torch.ones(1).to(device), atol=1e-6), "Expected embeddings to be L2 normalized"
print("DINO embedding is L2 normalized")

assert torch.all(embeds >= -1) and torch.all(embeds <= 1), "Expected embedding values to be in the range [-1, 1]"
print("DINO embedding values are in the range [-1, 1]")

dino_score = evaluator.dino_score([x], [x])
print("DINO score:", dino_score)
assert -1 <= dino_score <= 1, "Expected DINO score to be in the range [-1, 1]"
print("DINO score is in the range [-1, 1]")

dino_score = evaluator.dino_score(
    [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))],
    [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))]
)
print("DINO score:", dino_score)
assert -1 - 1e-6 <= dino_score <= 1 + 1e-6, "Expected DINO score to be in the range [-1, 1]"
print("DINO score is in the range [-1, 1]")

