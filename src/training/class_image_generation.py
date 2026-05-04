from __future__ import annotations

from pathlib import Path

import torch
try:
    import sdnq  # noqa: F401 — registers SDNQ quantization support into diffusers/transformers
except ImportError:
    pass

from diffusers import AutoPipelineForText2Image, Flux2KleinPipeline
from tqdm.auto import tqdm

from src.training.utils import count_images, QWEN3_CHAT_TEMPLATE


def _resolve_torch_dtype(mixed_precision: str | None) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def generate_class_images(
    pretrained_model_path: str,
    class_prompt: str,
    class_data_dir: str | Path,
    num_images_to_generate: int,
    batch_size: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int | None = None,
    mixed_precision: str | None = None,
    file_prefix: str = "class",
) -> int:
    if num_images_to_generate <= 0:
        return 0

    out_dir = Path(class_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_torch_dtype(mixed_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = AutoPipelineForText2Image.from_pretrained(
        pretrained_model_path,
        torch_dtype=dtype,
    ).to(device)
    if isinstance(pipe, Flux2KleinPipeline) and not getattr(pipe.tokenizer, "chat_template", None):
        pipe.tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
    pipe.set_progress_bar_config(disable=True)

    created = 0
    start_index = count_images(out_dir)
    num_batches = (num_images_to_generate + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_idx in tqdm(range(num_batches), desc="Generating class images"):
            current_batch_size = min(batch_size, num_images_to_generate - created)
            prompts = [class_prompt] * current_batch_size
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(
                    seed + start_index + batch_idx
                )

            images = pipe(
                prompt=prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images

            for image in images:
                image_index = start_index + created
                file_path = out_dir / f"{file_prefix}_{image_index:06d}.png"
                image.save(file_path)
                created += 1

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return created

