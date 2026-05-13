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
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    Flux2KleinPipeline,
    AutoPipelineForText2Image,
    DDPMScheduler,
)
import bitsandbytes as bnb

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.training.utils import (
    DreamBoothDataset,
    QWEN3_CHAT_TEMPLATE,
    collate_dreambooth_batch,
    compute_class_image_target,
    count_images,
    discover_images,
    load_yaml_config,
    merge_configs,
    pack_latents,
    prepare_latent_image_ids,
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


def _detect_model_family(pipeline) -> str:
    if isinstance(pipeline, Flux2KleinPipeline):
        return "flux2"
    if isinstance(pipeline, StableDiffusion3Pipeline):
        return "sd3"
    if isinstance(pipeline, StableDiffusionXLPipeline):
        return "sdxl"
    return "sd15"


def _is_flow_matching(model_family: str) -> bool:
    return model_family in {"sd3", "flux2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA DreamBooth fine-tuning.")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config file.")
    parser.add_argument("--override", type=str, default=None, help="Path to override YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.override:
        config = merge_configs(config, load_yaml_config(args.override))
    config = resolve_base_config_runtime_values(config)

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
    with_prior_preservation = get_config_value(config, "with_prior_preservation")
    train_text_encoder_lora = config.get("train_text_encoder_lora", False)
    instance_data_dir = get_config_value(config, "instance_data_dir")
    instance_prompt = get_config_value(config, "instance_prompt")

    class_data_dir = config.get("class_data_dir")
    class_prompt = config.get("class_prompt")
    if with_prior_preservation:
        class_data_dir = get_config_value(config, "class_data_dir")
        class_prompt = get_config_value(config, "class_prompt")
        # Mirror the model_name subdirectory that generate_class_images.py uses.
        model_name = get_config_value(config, "model_name")
        class_data_dir = str(Path(class_data_dir).parent / model_name / Path(class_data_dir).name)

        num_instance_images = len(discover_images(instance_data_dir))
        target_class_images = compute_class_image_target(
            num_instance_images=num_instance_images,
            class_images_per_instance=get_config_value(config, "class_images_per_instance"),
        )
        existing_class_images = count_images(class_data_dir)
        accelerator.print(
            f"Class image target: {target_class_images}, existing: {existing_class_images}."
        )
        if existing_class_images < target_class_images:
            accelerator.print(
                f"ERROR: {target_class_images - existing_class_images} class images missing "
                f"in {class_data_dir}.\n"
                f"Run: python -m src.scripts.generate_class_images --config <your_config>"
            )
            raise SystemExit(1)

    # --- Load pipeline and extract components ---
    _mp = get_config_value(config, "mixed_precision")
    _load_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_mp, torch.float32)
    pipeline = AutoPipelineForText2Image.from_pretrained(base_model_path, torch_dtype=_load_dtype)
    model_family = _detect_model_family(pipeline)
    accelerator.print(f"Detected model family: {model_family}")

    # Shared pre-computed embedding state (populated for non-SD1.5 models)
    instance_prompt_embeds = instance_pooled_embeds = None
    instance_txt_ids = None
    class_prompt_embeds = class_pooled_embeds = None
    class_txt_ids = None
    add_time_ids = None

    if model_family == "sdxl":
        tokenizer = pipeline.tokenizer
        backbone = pipeline.unet
        vae = pipeline.vae
        noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        text_encoder = pipeline.text_encoder  # kept for optional LoRA
        with torch.no_grad():
            instance_prompt_embeds, _, instance_pooled_embeds, _ = pipeline.encode_prompt(
                instance_prompt, instance_prompt
            )
            if with_prior_preservation:
                class_prompt_embeds, _, class_pooled_embeds, _ = pipeline.encode_prompt(
                    class_prompt, class_prompt
                )
        resolution = get_config_value(config, "resolution")
        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]], dtype=torch.float32
        )

    elif model_family == "sd3":
        tokenizer = pipeline.tokenizer
        backbone = pipeline.transformer
        vae = pipeline.vae
        noise_scheduler = pipeline.scheduler
        text_encoder = None
        with torch.no_grad():
            instance_prompt_embeds, _, instance_pooled_embeds, _ = pipeline.encode_prompt(
                instance_prompt, instance_prompt, instance_prompt
            )
            if with_prior_preservation:
                class_prompt_embeds, _, class_pooled_embeds, _ = pipeline.encode_prompt(
                    class_prompt, class_prompt, class_prompt
                )

    elif model_family == "flux2":
        tokenizer = pipeline.tokenizer
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
            pipeline.tokenizer = tokenizer
        backbone = pipeline.transformer
        vae = pipeline.vae
        noise_scheduler = pipeline.scheduler
        text_encoder = None
        with torch.no_grad():
            instance_prompt_embeds, instance_txt_ids = pipeline.encode_prompt(instance_prompt)
            if with_prior_preservation:
                class_prompt_embeds, class_txt_ids = pipeline.encode_prompt(class_prompt)

    else:  # sd15
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        backbone = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        vae = pipeline.vae
        noise_scheduler = cast(DDPMScheduler, DDPMScheduler.from_config(pipeline.scheduler.config))

    # Cache VAE scalars now — quantized models sometimes strip these from config.json.
    vae_shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    vae_scale = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)

    del pipeline
    torch.cuda.empty_cache()

    # --- Freeze / parameterize trainable components ---
    vae.requires_grad_(False)
    backbone.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)

    use_lora = config.get("lora_rank") is not None

    if use_lora:
        lora_rank = get_config_value(config, "lora_rank")
        lora_alpha = get_config_value(config, "lora_alpha")
        lora_dropout = get_config_value(config, "lora_dropout")

        # SD3/Flux2 JointTransformerBlocks have a second set of projections for the text stream.
        # Without these, only image-stream attention adapts and subject learning is very slow.
        if model_family in {"sd3", "flux2"}:
            _attn_targets = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "add_out_proj"]
        else:
            _attn_targets = ["to_q", "to_k", "to_v", "to_out.0"]

        backbone_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=_attn_targets,
        )
        backbone.add_adapter(backbone_lora_config)

        # Text encoder LoRA: only SD1.5 and SDXL (SD3/Flux2 text encoders are deleted above)
        trains_text_encoder = train_text_encoder_lora and text_encoder is not None
        if trains_text_encoder:
            text_lora_config = LoraConfig(
                r=get_config_value(config, "text_lora_rank"),
                lora_alpha=get_config_value(config, "text_lora_alpha"),
                lora_dropout=get_config_value(config, "text_lora_dropout"),
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            text_encoder.add_adapter(text_lora_config)

        # LoRA layers inherit the backbone dtype (may be fp16). Cast trainable params to fp32
        # so GradScaler (active under fp16 mixed_precision) can unscale their gradients.
        for param in backbone.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        if trains_text_encoder:
            for param in text_encoder.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)

    else:  # full fine-tuning
        backbone.requires_grad_(True)
        # bf16 weight resolution (~0.4%) swallows updates at lr=5e-6; train in fp32.
        backbone.to(torch.float32)
        # Text encoder training only for SD1.5; SDXL/SD3/Flux2 use pre-computed embeddings.
        trains_text_encoder = (
            config.get("train_text_encoder", False)
            and model_family == "sd15"
            and text_encoder is not None
        )
        if trains_text_encoder:
            text_encoder.requires_grad_(True)
            text_encoder.to(torch.float32)

    if config.get("gradient_checkpointing", False):
        backbone.enable_gradient_checkpointing()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    backbone.train()
    if text_encoder is not None:
        text_encoder.train()

    dataset = DreamBoothDataset(
        instance_data_dir=instance_data_dir,
        instance_prompt=instance_prompt,
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
    text_learning_rate = (
        config.get("text_learning_rate")
        or config.get("text_encoder_learning_rate")
        or learning_rate
    )
    adam_beta1 = get_config_value(config, "adam_beta1")
    adam_beta2 = get_config_value(config, "adam_beta2")
    adam_weight_decay = get_config_value(config, "adam_weight_decay")
    adam_epsilon = get_config_value(config, "adam_epsilon")

    params_to_optimize = [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": learning_rate}]
    if trains_text_encoder:
        params_to_optimize.append({
            "params": [p for p in text_encoder.parameters() if p.requires_grad],
            "lr": text_learning_rate,
        })

    use_8bit_adam = config.get("use_8bit_adam", False)
    if use_8bit_adam:
        optimizer = bnb.optim.AdamW8bit(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
    else:
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

    if trains_text_encoder:
        backbone, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            backbone, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        backbone, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            backbone, optimizer, train_dataloader, lr_scheduler
        )
        if text_encoder is not None:
            text_encoder = accelerator.prepare(text_encoder)
    vae = accelerator.prepare(vae)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move pre-computed embeddings to device
    device = accelerator.device
    if instance_prompt_embeds is not None:
        instance_prompt_embeds = instance_prompt_embeds.to(device=device, dtype=weight_dtype)
        if with_prior_preservation:
            class_prompt_embeds = class_prompt_embeds.to(device=device, dtype=weight_dtype)
    if instance_pooled_embeds is not None:
        instance_pooled_embeds = instance_pooled_embeds.to(device=device, dtype=weight_dtype)
        if with_prior_preservation:
            class_pooled_embeds = class_pooled_embeds.to(device=device, dtype=weight_dtype)
    if add_time_ids is not None:
        add_time_ids = add_time_ids.to(device=device, dtype=weight_dtype)
    if instance_txt_ids is not None:
        instance_txt_ids = instance_txt_ids.to(device=device, dtype=weight_dtype)
        if with_prior_preservation:
            class_txt_ids = class_txt_ids.to(device=device, dtype=weight_dtype)

    if _is_flow_matching(model_family):
        if getattr(noise_scheduler.config, "use_dynamic_shifting", False):
            # Flux2: timestep distribution is shifted based on image resolution.
            resolution = get_config_value(config, "resolution")
            image_seq_len = (resolution // 16) ** 2  # VAE 8× + packing 2× = 16× total
            base_seq_len = getattr(noise_scheduler.config, "base_image_seq_len", 256)
            max_seq_len = getattr(noise_scheduler.config, "max_image_seq_len", 4096)
            base_shift = getattr(noise_scheduler.config, "base_shift", 0.5)
            max_shift = getattr(noise_scheduler.config, "max_shift", 1.15)
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = image_seq_len * m + (base_shift - m * base_seq_len)
            noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=device, mu=mu)
        else:
            noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=device)

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    prior_loss_weight = config.get("prior_loss_weight", 1.0)
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
            with accelerator.accumulate(backbone):
                vae_dtype = next(vae.parameters()).dtype
                pixel_values = batch["pixel_values"].to(dtype=vae_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                latents = ((latents - vae_shift) * vae_scale).to(weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                if _is_flow_matching(model_family):
                    indices = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = noise_scheduler.timesteps[indices]
                else:
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                if model_family == "flux2":
                    latents_packed = pack_latents(latents)
                    noise_packed = pack_latents(noise)
                    noisy_latents = noise_scheduler.scale_noise(latents_packed, timesteps, noise_packed)
                    _, seq_h, seq_w = latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2
                    img_ids = prepare_latent_image_ids(bsz, seq_h, seq_w, device, weight_dtype)
                elif _is_flow_matching(model_family):  # sd3
                    noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)
                else:  # sd15, sdxl
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Assemble text conditioning
                if model_family == "sd15":
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                else:
                    # Expand pre-computed embeddings to match batch
                    half = bsz // 2 if with_prior_preservation else bsz
                    if with_prior_preservation:
                        enc = torch.cat([
                            instance_prompt_embeds.expand(half, -1, -1),
                            class_prompt_embeds.expand(half, -1, -1),
                        ])
                    else:
                        enc = instance_prompt_embeds.expand(bsz, -1, -1)
                    encoder_hidden_states = enc

                # Model forward
                if model_family == "sdxl":
                    if with_prior_preservation:
                        pooled = torch.cat([
                            instance_pooled_embeds.expand(half, -1),
                            class_pooled_embeds.expand(half, -1),
                        ])
                        time_ids = add_time_ids.expand(bsz, -1)
                    else:
                        pooled = instance_pooled_embeds.expand(bsz, -1)
                        time_ids = add_time_ids.expand(bsz, -1)
                    model_pred = backbone(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                        return_dict=False,
                    )[0]

                elif model_family == "sd3":
                    if with_prior_preservation:
                        pooled = torch.cat([
                            instance_pooled_embeds.expand(half, -1),
                            class_pooled_embeds.expand(half, -1),
                        ])
                    else:
                        pooled = instance_pooled_embeds.expand(bsz, -1)
                    model_pred = backbone(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled,
                        return_dict=False,
                    )[0]

                elif model_family == "flux2":
                    if with_prior_preservation:
                        txt_ids = torch.cat([
                            instance_txt_ids.expand(half, -1, -1),
                            class_txt_ids.expand(half, -1, -1),
                        ])
                    else:
                        txt_ids = instance_txt_ids.expand(bsz, -1, -1)
                    model_pred = backbone(
                        hidden_states=noisy_latents,
                        timestep=timesteps / 1000,
                        encoder_hidden_states=encoder_hidden_states,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]

                else:  # sd15
                    model_pred = backbone(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute target
                if model_family == "flux2":
                    target = noise_packed - latents_packed
                elif model_family == "sd3":
                    target = noise - latents
                else:
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
                    params = [p for p in backbone.parameters() if p.requires_grad]
                    if trains_text_encoder:
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
                    _unwrapped_te = accelerator.unwrap_model(text_encoder) if trains_text_encoder else None
                    if use_lora:
                        _save_lora(model_family, checkpoint_dir, accelerator.unwrap_model(backbone), _unwrapped_te)
                    else:
                        _save_full(model_family, checkpoint_dir, accelerator.unwrap_model(backbone), _unwrapped_te)

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
                    val_pipe = _load_val_pipeline(model_family, base_model_path, weight_dtype)
                    if use_lora:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
                        val_pipe.load_lora_weights(str(checkpoint_dir))
                    else:
                        _inject_backbone(
                            val_pipe, model_family, accelerator.unwrap_model(backbone),
                            text_encoder=accelerator.unwrap_model(text_encoder) if trains_text_encoder else None,
                        )
                        accelerator.unwrap_model(backbone).eval()
                    val_pipe = val_pipe.to(accelerator.device)
                    save_validation_images(
                        pipeline=val_pipe,
                        prompt=validation_prompt,
                        output_dir=output_dir / "validation",
                        step=global_step,
                        num_images=get_config_value(config, "num_validation_images"),
                        guidance_scale=get_config_value(config, "validation_guidance_scale"),
                        num_inference_steps=get_config_value(config, "validation_steps_infer"),
                        generator_seed=get_config_value(config, "seed"),
                    )
                    del val_pipe
                    torch.cuda.empty_cache()
                    if not use_lora:
                        accelerator.unwrap_model(backbone).train()

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
        _unwrapped_te = accelerator.unwrap_model(text_encoder) if trains_text_encoder else None
        if use_lora:
            _save_lora(model_family, final_dir, accelerator.unwrap_model(backbone), _unwrapped_te)
        else:
            _save_full(model_family, final_dir, accelerator.unwrap_model(backbone), _unwrapped_te)
        tokenizer.save_pretrained(final_dir / "tokenizer")

    accelerator.end_training()


def _save_full(model_family: str, save_dir: Path, backbone, text_encoder) -> None:
    subdir = "transformer" if model_family in {"sd3", "flux2"} else "unet"
    backbone.save_pretrained(save_dir / subdir)
    if text_encoder is not None:
        text_encoder.save_pretrained(save_dir / "text_encoder")


def _inject_backbone(pipeline, model_family: str, backbone, text_encoder=None) -> None:
    if model_family in {"sd3", "flux2"}:
        pipeline.transformer = backbone
    else:
        pipeline.unet = backbone
    if text_encoder is not None:
        pipeline.text_encoder = text_encoder


def _save_lora(model_family: str, save_dir: Path, backbone, text_encoder) -> None:
    backbone_layers = get_peft_model_state_dict(backbone)
    text_enc_layers = get_peft_model_state_dict(text_encoder) if text_encoder is not None else None

    if model_family == "sdxl":
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=save_dir,
            unet_lora_layers=backbone_layers,
            text_encoder_lora_layers=text_enc_layers,
        )
    elif model_family == "sd3":
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=backbone_layers,
        )
    elif model_family == "flux2":
        Flux2KleinPipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=backbone_layers,
        )
    else:  # sd15
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_dir,
            unet_lora_layers=backbone_layers,
            text_encoder_lora_layers=text_enc_layers,
        )


def _load_val_pipeline(model_family: str, base_model_path: str, weight_dtype: torch.dtype):
    if model_family == "sdxl":
        return StableDiffusionXLPipeline.from_pretrained(base_model_path, torch_dtype=weight_dtype)
    if model_family == "sd3":
        return StableDiffusion3Pipeline.from_pretrained(base_model_path, torch_dtype=weight_dtype)
    if model_family == "flux2":
        return Flux2KleinPipeline.from_pretrained(base_model_path, torch_dtype=weight_dtype)
    return StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
    )


if __name__ == "__main__":
    main()
