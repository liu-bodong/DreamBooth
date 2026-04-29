from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
#from diffusers.optimization import get_scheduler
#from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
#from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDPMScheduler, get_scheduler, StableDiffusionPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.training.utils import (
    DreamBoothDataset,
    collate_dreambooth_batch,
    load_yaml_config,
    save_validation_images,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full UNet DreamBooth fine-tuning.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in config and default is None:
        raise ValueError(f"Missing required config key: {key}")
    return config.get(key, default)


def main() -> None:

    args = parse_args()
    config = load_yaml_config(args.config)

    output_dir = Path(get_config_value(config, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=get_config_value(config, "gradient_accumulation_steps", 1),
        mixed_precision=get_config_value(config, "mixed_precision", "fp16"),
        log_with=get_config_value(config, "log_with", None),
        project_dir=str(output_dir / "logs"),
    )
    if get_config_value(config, "seed", None) is not None:
        set_seed(config["seed"])

    base_model_path = get_config_value(config, "pretrained_model_path")
    with_prior_preservation = get_config_value(config, "with_prior_preservation", True)
    train_text_encoder = get_config_value(config, "train_text_encoder", False)

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        requires_safety_checker=False,
    )
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = cast(DDPMScheduler, DDPMScheduler.from_config(pipeline.scheduler.config))
    del pipeline

    vae.requires_grad_(False)
    if train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    unet.train()

    dataset = DreamBoothDataset(
        instance_data_dir=get_config_value(config, "instance_data_dir"),
        instance_prompt=get_config_value(config, "instance_prompt"),
        class_data_dir=config.get("class_data_dir"),
        class_prompt=config.get("class_prompt"),
        image_size=get_config_value(config, "resolution", 512),
        center_crop=get_config_value(config, "center_crop", False),
    )

    if with_prior_preservation and not config.get("class_data_dir"):
        raise ValueError("class_data_dir is required when with_prior_preservation=True.")

    train_dataloader = DataLoader(
        dataset,
        batch_size=get_config_value(config, "train_batch_size", 1),
        shuffle=True,
        num_workers=get_config_value(config, "dataloader_num_workers", 4),
        collate_fn=lambda examples: collate_dreambooth_batch(
            examples,
            tokenizer=tokenizer,
            with_prior_preservation=with_prior_preservation,
        ),
    )

    learning_rate = get_config_value(config, "learning_rate", 1e-6)
    text_encoder_lr = get_config_value(config, "text_encoder_learning_rate", learning_rate)
    adam_beta1 = get_config_value(config, "adam_beta1", 0.9)
    adam_beta2 = get_config_value(config, "adam_beta2", 0.999)
    adam_weight_decay = get_config_value(config, "adam_weight_decay", 1e-2)
    adam_epsilon = get_config_value(config, "adam_epsilon", 1e-8)

    params_to_optimize = [{"params": unet.parameters(), "lr": learning_rate}]
    if train_text_encoder:
        params_to_optimize.append({"params": text_encoder.parameters(), "lr": text_encoder_lr})

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    max_train_steps = get_config_value(config, "max_train_steps", 1000)
    num_train_epochs = get_config_value(config, "num_train_epochs", None)
    if num_train_epochs is None:
        steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    else:
        max_train_steps = num_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.gradient_accumulation_steps
        )

    lr_scheduler = get_scheduler(
        name=get_config_value(config, "lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=get_config_value(config, "lr_warmup_steps", 0),
        num_training_steps=max_train_steps,
    )

    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        text_encoder = accelerator.prepare(text_encoder)
    vae = accelerator.prepare(vae)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    prior_loss_weight = get_config_value(config, "prior_loss_weight", 1.0)
    max_grad_norm = get_config_value(config, "max_grad_norm", 1.0)
    checkpointing_steps = get_config_value(config, "checkpointing_steps", 500)
    validation_steps = get_config_value(config, "validation_steps", 500)
    validation_prompt = get_config_value(config, "validation_prompt", None)

    for _ in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config["num_train_timesteps"],
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.int32,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config["prediction_type"] == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if with_prior_preservation:
                    model_pred_instance, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target_instance, target_prior = torch.chunk(target, 2, dim=0)
                    instance_loss = F.mse_loss(model_pred_instance.float(), target_instance.float())
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float())
                    loss = instance_loss + prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params = unet.parameters() if not train_text_encoder else list(unet.parameters()) + list(text_encoder.parameters())
                    accelerator.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % checkpointing_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(unet).save_pretrained(checkpoint_dir / "unet")
                    if train_text_encoder:
                        accelerator.unwrap_model(text_encoder).save_pretrained(
                            checkpoint_dir / "text_encoder"
                        )

                if (
                    accelerator.is_main_process
                    and validation_prompt
                    and global_step % validation_steps == 0
                ):
                    val_pipe = StableDiffusionPipeline.from_pretrained(
                        base_model_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        safety_checker=None,
                        requires_safety_checker=False,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                    save_validation_images(
                        pipeline=val_pipe,
                        prompt=validation_prompt,
                        output_dir=output_dir / "validation",
                        step=global_step,
                        num_images=get_config_value(config, "num_validation_images", 4),
                        guidance_scale=get_config_value(config, "validation_guidance_scale", 7.5),
                        num_inference_steps=get_config_value(config, "validation_steps_infer", 30),
                        generator_seed=get_config_value(config, "seed", None),
                    )
                    del val_pipe

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(final_dir / "unet")
        if train_text_encoder:
            accelerator.unwrap_model(text_encoder).save_pretrained(final_dir / "text_encoder")
        tokenizer.save_pretrained(final_dir / "tokenizer")

    accelerator.end_training()


if __name__ == "__main__":
    main()
