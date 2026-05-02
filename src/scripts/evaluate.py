from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.evaluation.CLIP import CLIPEvaluator
from src.evaluation.DINO import DINOEvaluator
from src.evaluation.diversity import DiversityEvaluator
from src.training.utils import IMAGE_EXTENSIONS


def load_images(directory: str | Path) -> list[Image.Image]:
    root = Path(directory)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {root}")
    return [Image.open(p).convert("RGB") for p in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DreamBooth generated images.")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used to generate images (for CLIP-T).")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    generated = load_images(args.generated_dir)
    real = load_images(args.real_dir)
    print(f"Loaded {len(generated)} generated and {len(real)} real images.")

    dino = DINOEvaluator(device)
    clip = CLIPEvaluator(device)
    div = DiversityEvaluator(device)

    dino_matrix = dino.dino_score(generated, real)
    clip_i_matrix = clip.clip_i_score(generated, real)
    clip_t_vector = clip.clip_t_score(generated, [args.prompt] * len(generated))
    div_scores = div.diversity_score(generated)

    print("\n=== Evaluation Results ===")
    print(f"DINO score:   {dino_matrix.mean().item():.4f}")
    print(f"CLIP-I score: {clip_i_matrix.mean().item():.4f}")
    print(f"CLIP-T score: {clip_t_vector.mean().item():.4f}")
    if div_scores.numel() > 0:
        print(f"DIV score:    {div_scores.mean().item():.4f}")
    else:
        print("DIV score:    N/A (need >= 2 images)")


if __name__ == "__main__":
    main()
