from diffusers import (
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
    UNet2DConditionModel,
    SD3Transformer2DModel,
)

import torch
from pathlib import Path

def pick_models_root() -> Path:
    default_root = (Path(__file__).resolve().parents[2] / "base_models").resolve()
    raw = input(f"Models directory [{default_root}]: ").strip()
    root = Path(raw).expanduser().resolve() if raw else default_root
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid models directory: {root}")
    return root

def choose_from_menu(items: list[str], prompt: str) -> int:
    if not items:
        raise ValueError("No options found.")
    print()
    for i, item in enumerate(items, start=1):
        print(f"{i}. {item}")
    raw = input(f"{prompt} [1-{len(items)}]: ").strip()
    idx = int(raw) - 1 if raw else 0
    if idx < 0 or idx >= len(items):
        raise ValueError("Invalid menu choice.")
    return idx

def discover_model_roots(root: Path) -> list[Path]:
    roots = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or ".cache" in p.parts:
            continue
        if (p / "structure").is_dir():
            roots.append(p)
    return roots

def find_quant_files(model_root: Path) -> list[Path]:
    quant_dir = model_root / "quant_weights"
    if not quant_dir.is_dir():
        return []
    files = []
    files.extend(sorted(quant_dir.glob("*.gguf")))
    files.extend(sorted(quant_dir.glob("*.safetensors")))
    return files

def has_structure_weights(structure_dir: Path) -> bool:
    return any(structure_dir.rglob("*.safetensors")) or any(structure_dir.rglob("*.bin"))

def load_model(structure_path: Path, loader_key: str):
    dtype = torch.bfloat16
    if loader_key == "flux2":
        return Flux2Transformer2DModel.from_pretrained(
            str(structure_path), subfolder="transformer", torch_dtype=dtype
        )
    if loader_key == "flux1":
        return FluxTransformer2DModel.from_pretrained(
            str(structure_path), subfolder="transformer", torch_dtype=dtype
        )
    if loader_key == "sd3":
        return SD3Transformer2DModel.from_pretrained(
            str(structure_path), subfolder="transformer", torch_dtype=dtype
        )
    return UNet2DConditionModel.from_pretrained(
        str(structure_path), subfolder="unet", torch_dtype=dtype
    )

root = pick_models_root()
model_roots = discover_model_roots(root)
model_idx = choose_from_menu([p.name for p in model_roots], "Select model")
model_root = model_roots[model_idx]
structure_path = model_root / "structure"

quant_files = find_quant_files(model_root)
mode_choices = ["structure only"]
if quant_files:
    mode_choices.append("structure + quantized weights")
mode_idx = choose_from_menu(mode_choices, "Select mode")

selected_quant = None
if mode_choices[mode_idx].startswith("structure +"):
    q_idx = choose_from_menu([p.name for p in quant_files], "Select quant file")
    selected_quant = quant_files[q_idx]

loader_choices = [
    "flux2 (transformer)",
    "flux1 (transformer)",
    "sd3 (transformer)",
    "unet (unet)",
]
loader_idx = choose_from_menu(loader_choices, "Select loader type")
loader_key = ["flux2", "flux1", "sd3", "unet"][loader_idx]

print(f"\nModel root: {model_root}")
print(f"Structure path: {structure_path}")
print(f"Loader: {loader_key}")
if selected_quant:
    size_gb = selected_quant.stat().st_size / 1e9
    print(f"Quant weights: {selected_quant.name} ({size_gb:.3f} GB)")

if not has_structure_weights(structure_path):
    print("\nNo full weights found under structure/.")
    print("Cannot compute num_parameters/get_memory_footprint from diffusers module.")
    exit(1)

model = load_model(structure_path, loader_key)
params = model.num_parameters()
vram_bytes = model.get_memory_footprint()
print(f"Total Parameters: {params / 1e9:.4f} B")
print(f"Base VRAM Footprint: {vram_bytes / 1e9:.4f} GB")