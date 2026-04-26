from diffusers import (
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
    UNet2DConditionModel,
    SD3Transformer2DModel,
)
import torch

model_path = (
    "/home/liu/dev/school/DreamBooth/base_models/black-forest-labs_FLUX.2-klein-4B"
)
# 1. Load the model
model = Flux2Transformer2DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# 2. Ask it for the exact parameter count
params = model.num_parameters()
print(f"Total Parameters: {params / 1e9:.4f} B")

# 3. Ask it for the exact VRAM footprint (Weights only)
vram_bytes = model.get_memory_footprint()
print(f"Base VRAM Footprint: {vram_bytes / 1e9:.4f} GB")
