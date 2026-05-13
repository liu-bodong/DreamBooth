from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel
from transformers import CLIPTextModel

from src.training.utils import get_config_value, load_yaml_config, merge_configs, resolve_base_config_runtime_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation images for all three DreamBooth eval modes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_prompts", type=str, default="configs/eval_prompts.yaml")
    parser.add_argument("--modes", type=str, default="class,instance,extended",
                        help="Comma-separated list of modes to run: class, instance, extended.")
    parser.add_argument("--num_images", type=int, default=12,
                        help="Images per prompt for class and instance modes.")
    parser.add_argument("--num_images_extended", type=int, default=4,
                        help="Images per prompt for extended mode.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    return parser.parse_args()


def _generate_and_save(
    pipeline: AutoPipelineForText2Image,
    prompt: str,
    n: int,
    output_dir: Path,
    seed: int | None,
    cfg: float,
    num_steps: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    images = pipeline(
        prompt=[prompt] * n,
        guidance_scale=cfg,
        num_inference_steps=num_steps,
        generator=generator,
    ).images
    for i, image in enumerate(images):
        path = output_dir / f"generated_{i:04d}.png"
        image.save(path)
    print(f"[{output_dir.name}] Saved {len(images)} images → {output_dir}")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.override:
        config = merge_configs(config, load_yaml_config(args.override))
    config = resolve_base_config_runtime_values(config)

    class_name = config["class_name"]
    unique_token = config.get("unique_token")
    class_prompt = config["class_prompt"]
    instance_prompt = config["instance_prompt"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = config.get("mixed_precision", "no")
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision, torch.float32)

    base_model_path = get_config_value(config, "pretrained_model_path")
    checkpoint_path = Path(args.checkpoint)
    cfg = args.cfg if args.cfg is not None else get_config_value(config, "validation_guidance_scale")
    num_steps = args.num_steps if args.num_steps is not None else get_config_value(config, "validation_steps_infer")

    if config.get("lora_rank") is not None:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model_path, torch_dtype=weight_dtype
        ).to(device)
        pipeline.load_lora_weights(str(checkpoint_path))
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model_path, torch_dtype=weight_dtype
        ).to(device)
        unet_dir = checkpoint_path / "unet"
        transformer_dir = checkpoint_path / "transformer"
        if unet_dir.exists():
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                unet_dir, torch_dtype=weight_dtype
            ).to(device)
        elif transformer_dir.exists():
            transformer_cls = type(pipeline.transformer)
            pipeline.transformer = transformer_cls.from_pretrained(
                transformer_dir, torch_dtype=weight_dtype
            ).to(device)
        text_encoder_path = checkpoint_path / "text_encoder"
        if text_encoder_path.exists():
            pipeline.text_encoder = CLIPTextModel.from_pretrained(
                text_encoder_path, torch_dtype=weight_dtype
            ).to(device)

    output_dir = Path(args.output_dir)
    modes = [m.strip() for m in args.modes.split(",")]

    if "class" in modes:
        print(f"Class prompt: {class_prompt!r}")
        _generate_and_save(pipeline, class_prompt, args.num_images, output_dir / "class",
                           args.seed, cfg, num_steps, device)

    if "instance" in modes:
        print(f"Instance prompt: {instance_prompt!r}")
        _generate_and_save(pipeline, instance_prompt, args.num_images, output_dir / "instance",
                           args.seed, cfg, num_steps, device)

    if "extended" in modes:
        eval_prompts_cfg = load_yaml_config(args.eval_prompts)
        for entry in eval_prompts_cfg["extended_prompts"]:
            prompt = entry["template"].format(unique_token=unique_token, class_name=class_name)
            print(f"Extended prompt [{entry['tag']}]: {prompt!r}")
            _generate_and_save(pipeline, prompt, args.num_images_extended,
                               output_dir / "extended" / entry["tag"],
                               args.seed, cfg, num_steps, device)


if __name__ == "__main__":
    main()
