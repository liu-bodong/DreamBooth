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
        return max(len(self.instance_images), len(self.class_images) if self.class_images else 0)
    
    def _preprocess_image(self, path: Path, mode: str, width: int, height: int) -> torch.Tensor:
        image = Image.open(path).convert(mode)
        # check if image resolution matches desired resolution
        if image.size != (width, height):
            if mode == "upsample":
                image = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC)(image)
            else:
                raise ValueError(f"Image resolution does not match desired resolution: {image.size} != ({width}, {height})")
        # convert to tensor
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(image.size[1], image.size[0], len(image.getbands())).float()
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
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

    with torch.no_grad():
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
