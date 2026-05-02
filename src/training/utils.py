from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return {**base, **override}


def discover_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Image directory not found: {root}")

    images = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in: {root}")
    return images


def count_images(image_dir: str | Path) -> int:
    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def compute_class_image_target(
    num_instance_images: int,
    class_images_per_instance: int,
) -> int:
    if num_instance_images <= 0:
        raise ValueError("num_instance_images must be > 0.")
    if class_images_per_instance <= 0:
        raise ValueError("class_images_per_instance must be > 0.")
    return class_images_per_instance * num_instance_images


def get_config_value(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    return config.get(key)


def resolve_base_config_runtime_values(
    config: dict[str, Any],
    subject_override: str | None = None,
) -> dict[str, Any]:
    resolved = dict(config)

    subject_name = subject_override or resolved.get("subject_name")
    if not isinstance(subject_name, str) or not subject_name:
        raise ValueError("Missing subject_name. Set it in base config or pass --subject.")
    resolved["subject_name"] = subject_name

    instance_data_dir = resolved.get("instance_data_dir")
    if not instance_data_dir:
        instance_root = resolved.get("instance_data_root", "data/processed/instance")
        resolved["instance_data_dir"] = str(Path(str(instance_root)) / subject_name)

    class_name = resolved.get("class_name")
    if not class_name:
        subject_class_map = resolved.get("subject_class_map", {})
        if isinstance(subject_class_map, dict):
            class_name = subject_class_map.get(subject_name)
    if not isinstance(class_name, str) or not class_name:
        class_name = subject_name
    resolved["class_name"] = class_name

    class_data_dir = resolved.get("class_data_dir")
    if not class_data_dir:
        class_root = resolved.get("class_data_root", "data/processed/class")
        class_dir_name = class_name.replace(" ", "_")
        resolved["class_data_dir"] = str(Path(str(class_root)) / class_dir_name)

    unique_token = resolved.get("unique_token")
    if unique_token is not None and (not isinstance(unique_token, str) or not unique_token):
        raise ValueError("unique_token must be a non-empty string when provided.")
    if unique_token is not None:
        resolved["unique_token"] = unique_token

    prompt_templates: dict[str, Any] = {}
    prompt_templates_path = resolved.get("prompt_templates_path", "configs/prompt_templates.yaml")
    if isinstance(prompt_templates_path, str) and Path(prompt_templates_path).exists():
        loaded_templates = load_yaml_config(prompt_templates_path)
        if isinstance(loaded_templates, dict):
            prompt_templates = loaded_templates

    if not resolved.get("instance_prompt"):
        token_for_prompt = resolved.get("unique_token")
        if not isinstance(token_for_prompt, str) or not token_for_prompt:
            raise ValueError(
                "Missing unique_token for prompt generation. Set unique_token or set instance_prompt."
            )
        template = str(
            resolved.get(
                "instance_prompt_template",
                prompt_templates.get("instance_prompt_template", "a photo of {unique_token} {class_name}"),
            )
        )
        resolved["instance_prompt"] = template.format(
            unique_token=token_for_prompt,
            class_name=class_name,
        )

    if not resolved.get("class_prompt"):
        template = str(
            resolved.get(
                "class_prompt_template",
                prompt_templates.get("class_prompt_template", "a photo of {class_name}"),
            )
        )
        resolved["class_prompt"] = template.format(class_name=class_name)

    if not resolved.get("validation_prompt"):
        template = str(
            resolved.get(
                "validation_prompt_template",
                prompt_templates.get(
                    "validation_prompt_template",
                    resolved.get(
                        "instance_prompt_template",
                        prompt_templates.get("instance_prompt_template", "a photo of {unique_token} {class_name}"),
                    ),
                ),
            )
        )
        token_for_prompt = resolved.get("unique_token")
        if "{unique_token}" in template:
            if not isinstance(token_for_prompt, str) or not token_for_prompt:
                raise ValueError(
                    "Missing unique_token for validation_prompt generation. Set unique_token or validation_prompt."
                )
            resolved["validation_prompt"] = template.format(
                unique_token=token_for_prompt,
                class_name=class_name,
            )
        else:
            resolved["validation_prompt"] = template.format(class_name=class_name)

    return resolved

@dataclass(frozen=True)
class PromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None


def tokenize_prompts(tokenizer, prompts: list[str], max_length: int | None = None) -> PromptBatch:
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length or tokenizer.model_max_length,
        return_tensors="pt",
    )
    return PromptBatch(
        input_ids=tokenized.input_ids,
        attention_mask=getattr(tokenized, "attention_mask", None),
    )


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_dir: str | Path,
        instance_prompt: str,
        image_size: int,
        class_data_dir: str | Path | None = None,
        class_prompt: str | None = None,
        center_crop: bool = False,
    ) -> None:
        self.instance_images = discover_images(instance_data_dir)
        self.instance_prompt = instance_prompt
        self.class_images = discover_images(class_data_dir) if class_data_dir else []
        self.class_prompt = class_prompt
        self.image_size = image_size
        self.center_crop = center_crop

        if self.class_images and not class_prompt:
            raise ValueError("class_prompt is required when class_data_dir is provided.")

    def __len__(self) -> int:
        if self.class_images:
            return max(len(self.instance_images), len(self.class_images))
        return len(self.instance_images)

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.center_crop:
            min_size = min(image.size)
            left = (image.width - min_size) // 2
            top = (image.height - min_size) // 2
            image = image.crop((left, top, left + min_size, top + min_size))
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tensor)
        return tensor.contiguous()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}

        inst_path = self.instance_images[idx % len(self.instance_images)]
        item["instance_pixel_values"] = self._preprocess_image(inst_path)
        item["instance_prompt"] = self.instance_prompt

        if self.class_images:
            class_path = self.class_images[idx % len(self.class_images)]
            item["class_pixel_values"] = self._preprocess_image(class_path)
            item["class_prompt"] = self.class_prompt

        return item


def collate_dreambooth_batch(
    examples: list[dict[str, Any]],
    tokenizer,
    with_prior_preservation: bool,
) -> dict[str, torch.Tensor]:
    pixel_values = [example["instance_pixel_values"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    if with_prior_preservation:
        class_pixels = [example["class_pixel_values"] for example in examples]
        class_prompts = [example["class_prompt"] for example in examples]
        pixel_values.extend(class_pixels)
        prompts.extend(class_prompts)

    tokenized = tokenize_prompts(tokenizer, prompts)
    batch = {
        "pixel_values": torch.stack(pixel_values).float(),
        "input_ids": tokenized.input_ids,
    }
    if tokenized.attention_mask is not None:
        batch["attention_mask"] = tokenized.attention_mask
    return batch


def save_validation_images(
    pipeline,
    prompt: str,
    output_dir: str | Path,
    step: int,
    num_images: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    generator_seed: int | None = None,
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_paths: list[Path] = []

    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(generator_seed + step)

    unet_dtype = next(pipeline.unet.parameters()).dtype
    device_type = pipeline.device.type
    use_autocast = device_type in {"cuda", "mps"} and unet_dtype in {torch.float16, torch.bfloat16}
    vae_dtype = next(pipeline.vae.parameters()).dtype
    if vae_dtype in {torch.float16, torch.bfloat16}:
        # Decode-time dtype mismatches can happen on some stacks; use fp32 VAE for stable validation.
        pipeline.vae.to(dtype=torch.float32)

    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type=device_type, dtype=unet_dtype):
                images = pipeline(
                    prompt=[prompt] * num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images
        else:
            images = pipeline(
                prompt=[prompt] * num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images

    for index, image in enumerate(images):
        file_path = out_dir / f"step_{step:06d}_{index:02d}.png"
        image.save(file_path)
        save_paths.append(file_path)

    return save_paths
