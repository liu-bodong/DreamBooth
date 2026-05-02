from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from src.training.utils import get_config_value, load_yaml_config, resolve_base_config_runtime_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a DreamBooth LoRA checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint directory.")
    parser.add_argument("--prompt", type=str, default=None, help="Override the prompt from config.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_base_config_runtime_values(load_yaml_config(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixed_precision = config.get("mixed_precision", "no")
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision, torch.float32)

    base_model_path = get_config_value(config, "pretrained_model_path")
    prompt = args.prompt or get_config_value(config, "instance_prompt")

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
    ).to(device)
    pipeline.load_lora_weights(args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    images = pipeline(
        prompt=[prompt] * args.num_images,
        guidance_scale=get_config_value(config, "validation_guidance_scale"),
        num_inference_steps=get_config_value(config, "validation_steps_infer"),
        generator=generator,
    ).images

    for i, image in enumerate(images):
        path = output_dir / f"generated_{i:04d}.png"
        image.save(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
