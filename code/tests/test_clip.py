import torch
import torchvision
from torchvision import transforms
import PIL.Image as Image
import pathlib
import numpy as np

from src.evaluation.CLIP import CLIPEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# x = transforms.ToTensor()(x).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

evaluator = CLIPEvaluator(device)

embeds = evaluator.embed([x])
print("CLIP embedding shape:", embeds.shape)
assert embeds.shape == (1, 512), "Expected embedding shape to be (1, 512)"

assert torch.allclose(embeds.norm(dim=1), torch.ones(1).to(device), atol=1e-6), "Expected embeddings to be L2 normalized"
print("CLIP embedding is L2 normalized")

assert torch.all(embeds >= -1) and torch.all(embeds <= 1), "Expected embedding values to be in the range [-1, 1]"
print("CLIP embedding values are in the range [-1, 1]")

clip_score = evaluator.clip_i_score([x], [x])
print("CLIP score:", clip_score)
assert -1 - 1e-6 <= clip_score <= 1 + 1e-6, "Expected CLIP score to be in the range [-1, 1]"
print("CLIP score is in the range [-1, 1]")

clip_score = evaluator.clip_i_score(
    [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))],
    [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))]
)
print("CLIP score:", clip_score)
assert -1 - 1e-6 <= clip_score <= 1 + 1e-6, "Expected CLIP score to be in the range [-1, 1]"
print("CLIP score is in the range [-1, 1]")

clip_t_score = evaluator.clip_t_score(
    [x],
    ["a random image"]
)
print("CLIP text-image score:", clip_t_score)
assert -1 - 1e-6 <= clip_t_score <= 1 + 1e-6, "Expected CLIP text-image score to be in the range [-1, 1]"
print("CLIP text-image score is in the range [-1, 1]")

