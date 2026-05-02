from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.training.class_image_generation import generate_class_images
from src.training.utils import (
    DreamBoothDataset,
    collate_dreambooth_batch,
    compute_class_image_target,
    count_images,
    discover_images,
    load_yaml_config,
    save_validation_images,
    get_config_value,
    resolve_base_config_runtime_values,
)

def should_run_event(
    global_step: int,
    base_every: int,
    tail_start_step: int | None,
    tail_every: int | None,
) -> bool:
    if tail_start_step is not None and tail_every is not None and global_step >= tail_start_step:
        return global_step % tail_every == 0
    return global_step % base_every == 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA DreamBooth fine-tuning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolve_base_config_runtime_values(load_yaml_config(args.config))

    output_dir = Path(get_config_value(config, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=get_config_value(config, "gradient_accumulation_steps"),
        mixed_precision=get_config_value(config, "mixed_precision"),
        log_with=get_config_value(config, "log_with"),
        project_dir=str(output_dir / "logs"),
    )
    if get_config_value(config, "seed") is not None:
        set_seed(config["seed"])

    base_model_path = get_config_value(config, "pretrained_model_path")
    
    # PPS is replaced by lora so default is False
    with_prior_preservation = get_config_value(config, "with_prior_preservation")
    train_text_encoder_lora = get_config_value(config, "train_text_encoder_lora")
    instance_data_dir = get_config_value(config, "instance_data_dir")

    class_data_dir = config.get("class_data_dir")
    class_prompt = config.get("class_prompt")
    if with_prior_preservation:
        class_data_dir = get_config_value(config, "class_data_dir")
        class_prompt = get_config_value(config, "class_prompt")
        Path(class_data_dir).mkdir(parents=True, exist_ok=True)

        num_instance_images = len(discover_images(instance_data_dir))
        target_class_images = compute_class_image_target(
            num_instance_images=num_instance_images,
            class_images_per_instance=get_config_value(config, "class_images_per_instance"),
        )
        existing_class_images = count_images(class_data_dir)
        missing_class_images = max(target_class_images - existing_class_images, 0)
        class_stem = Path(class_data_dir).name.replace(" ", "_")
        file_prefix = f"class_{class_stem}" if class_stem else "class"
        accelerator.print(
            f"Class image target: {target_class_images}, existing: {existing_class_images}, "
            f"missing: {missing_class_images}."
        )
        if missing_class_images > 0 and accelerator.is_main_process:
            generate_class_images(
                pretrained_model_path=base_model_path,
                class_prompt=class_prompt,
                class_data_dir=class_data_dir,
                num_images_to_generate=missing_class_images,
                batch_size=get_config_value(config, "class_gen_batch_size"),
                guidance_scale=get_config_value(config, "class_gen_guidance_scale"),
                num_inference_steps=get_config_value(config, "class_gen_num_inference_steps"),
                seed=config.get("seed"),
                mixed_precision=get_config_value(config, "mixed_precision"),
                file_prefix=file_prefix,
            )
        accelerator.wait_for_everyone()

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    
    # check if model has unet or transformer, and raise error if neither
    if hasattr(pipeline, "unet"):
        unet = pipeline.unet
    elif hasattr(pipeline, "transformer"):
        unet = pipeline.transformer
    else:
        raise ValueError("The provided pipeline does not have a UNet or Transformer.")
    
    noise_scheduler = cast(DDPMScheduler, DDPMScheduler.from_config(pipeline.scheduler.config))
    del pipeline

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_rank = get_config_value(config, "lora_rank")
    lora_alpha = get_config_value(config, "lora_alpha")
    lora_dropout = get_config_value(config, "lora_dropout")

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if train_text_encoder_lora:
        text_lora_config = LoraConfig(
            r=get_config_value(config, "text_lora_rank"),
            lora_alpha=get_config_value(config, "text_lora_alpha"),
            lora_dropout=get_config_value(config, "text_lora_dropout"),
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    unet.train()
    text_encoder.train()

    dataset = DreamBoothDataset(
        instance_data_dir=instance_data_dir,
        instance_prompt=get_config_value(config, "instance_prompt"),
        class_data_dir=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt if with_prior_preservation else None,
        image_size=get_config_value(config, "resolution"),
        center_crop=get_config_value(config, "center_crop"),
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=get_config_value(config, "train_batch_size"),
        shuffle=True,
        num_workers=get_config_value(config, "dataloader_num_workers"),
        collate_fn=lambda examples: collate_dreambooth_batch(
            examples,
            tokenizer=tokenizer,
            with_prior_preservation=with_prior_preservation,
        ),
    )

    learning_rate = get_config_value(config, "learning_rate")
    text_learning_rate = get_config_value(config, "text_learning_rate")
    adam_beta1 = get_config_value(config, "adam_beta1")
    adam_beta2 = get_config_value(config, "adam_beta2")
    adam_weight_decay = get_config_value(config, "adam_weight_decay")
    adam_epsilon = get_config_value(config, "adam_epsilon")

    params_to_optimize = [{"params": [p for p in unet.parameters() if p.requires_grad], "lr": learning_rate}]
    if train_text_encoder_lora:
        params_to_optimize.append(
            {
                "params": [p for p in text_encoder.parameters() if p.requires_grad],
                "lr": text_learning_rate,
            }
        )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    max_train_steps = get_config_value(config, "max_train_steps")
    steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=get_config_value(config, "lr_scheduler"),
        optimizer=optimizer,
        num_warmup_steps=get_config_value(config, "lr_warmup_steps"),
        num_training_steps=max_train_steps,
    )

    if train_text_encoder_lora:
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

    prior_loss_weight = get_config_value(config, "prior_loss_weight")
    max_grad_norm = get_config_value(config, "max_grad_norm")
    checkpointing_steps = get_config_value(config, "checkpointing_steps")
    validation_steps = get_config_value(config, "validation_steps")
    validation_prompt = get_config_value(config, "validation_prompt")
    tail_checkpointing_start_step = config.get("tail_checkpointing_start_step")
    tail_checkpointing_steps = config.get("tail_checkpointing_steps")
    tail_validation_start_step = config.get("tail_validation_start_step")
    tail_validation_steps = config.get("tail_validation_steps")

    for _ in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                vae_dtype = next(vae.parameters()).dtype
                pixel_values = batch["pixel_values"].to(dtype=vae_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config["num_train_timesteps"],
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
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
                    params = [p for p in unet.parameters() if p.requires_grad]
                    if train_text_encoder_lora:
                        params += [p for p in text_encoder.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and should_run_event(
                    global_step=global_step,
                    base_every=checkpointing_steps,
                    tail_start_step=tail_checkpointing_start_step,
                    tail_every=tail_checkpointing_steps,
                ):
                    checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=checkpoint_dir,
                        unet_lora_layers=get_peft_model_state_dict(accelerator.unwrap_model(unet)),
                        text_encoder_lora_layers=(
                            get_peft_model_state_dict(accelerator.unwrap_model(text_encoder))
                            if train_text_encoder_lora
                            else None
                        ),
                    )

                if (
                    accelerator.is_main_process
                    and validation_prompt
                    and should_run_event(
                        global_step=global_step,
                        base_every=validation_steps,
                        tail_start_step=tail_validation_start_step,
                        tail_every=tail_validation_steps,
                    )
                ):
                    val_pipe = StableDiffusionPipeline.from_pretrained(
                        base_model_path,
                        safety_checker=None,
                        requires_safety_checker=False,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                    val_pipe.load_lora_weights(
                        output_dir / f"checkpoint-{global_step:06d}",
                    )
                    save_validation_images(
                        pipeline=val_pipe,
                        prompt=validation_prompt,
                        output_dir=output_dir / "validation",
                        step=global_step,
                        num_images=get_config_value(config, "num_validation_images"),
                        guidance_scale=get_config_value(config, "validation_guidance_scale"),
                        num_inference_steps=get_config_value(config, "validation_steps_infer"),
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
        StableDiffusionPipeline.save_lora_weights(
            save_directory=final_dir,
            unet_lora_layers=get_peft_model_state_dict(accelerator.unwrap_model(unet)),
            text_encoder_lora_layers=(
                get_peft_model_state_dict(accelerator.unwrap_model(text_encoder))
                if train_text_encoder_lora
                else None
            ),
        )
        tokenizer.save_pretrained(final_dir / "tokenizer")

    accelerator.end_training()


if __name__ == "__main__":
    main()
