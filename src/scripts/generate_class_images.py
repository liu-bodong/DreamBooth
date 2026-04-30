from __future__ import annotations

import argparse
from pathlib import Path

from src.training.class_image_generation import generate_class_images
from src.training.utils import (
    compute_class_image_target,
    count_images,
    discover_images,
    get_config_value,
    load_yaml_config,
    resolve_base_config_runtime_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DreamBooth class images for prior preservation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_base_config_runtime_values(load_yaml_config(args.config))

    instance_data_dir = get_config_value(config, "instance_data_dir")
    class_data_dir = get_config_value(config, "class_data_dir")
    class_prompt = get_config_value(config, "class_prompt")
    pretrained_model_path = get_config_value(config, "pretrained_model_path")
    Path(class_data_dir).mkdir(parents=True, exist_ok=True)

    num_instance_images = len(discover_images(instance_data_dir))
    target_count = compute_class_image_target(
        num_instance_images=num_instance_images,
        class_images_per_instance=get_config_value(config, "class_images_per_instance"),
    )
    current_count = count_images(class_data_dir)
    missing_count = max(target_count - current_count, 0)
    class_stem = Path(class_data_dir).name.replace(" ", "_")
    file_prefix = f"class_{class_stem}" if class_stem else "class"

    print(
        f"Instance images: {num_instance_images}, class target: {target_count}, "
        f"existing class images: {current_count}, generating: {missing_count}."
    )

    generated = generate_class_images(
        pretrained_model_path=pretrained_model_path,
        class_prompt=class_prompt,
        class_data_dir=class_data_dir,
        num_images_to_generate=missing_count,
        batch_size=get_config_value(config, "class_gen_batch_size"),
        guidance_scale=get_config_value(config, "class_gen_guidance_scale"),
        num_inference_steps=get_config_value(config, "class_gen_num_inference_steps"),
        seed=config.get("seed"),
        mixed_precision=config.get("mixed_precision"),
        file_prefix=file_prefix,
    )
    print(f"Generated {generated} class images.")


if __name__ == "__main__":
    main()

