# THIS IS AN OLDER VERSION, USE `model_size_hf.py`
import torch
from thop import profile, clever_format
from diffusers import UNet2DConditionModel
from diffusers import Flux2Transformer2DModel


class SDXLUNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        # Pack the extra SDXL conditions back into a dictionary
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )


class SD15UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        return output.sample


class FLux2TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer, height=1024, width=1024):
        super().__init__()
        self.transformer = transformer

        self.img_ids = self._prepare_ids(height, width)

    def forward(self, latent_image, timestep, encoder_hidden_states):
        hidden_states = self._patch_image(latent_image)
        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        return output.sample


# 2. Point to your local SDXL folder
# model_path = "./base_models/stable-diffusion-v1-5_stable-diffusion-v1-5"
model_path = (
    "/home/liu/dev/school/DreamBooth/base_models/black-forest-labs_FLUX.2-klein-4B"
)

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the base UNet
    # base_unet = UNet2DConditionModel.from_pretrained(
    #     model_path,
    #     subfolder="unet",
    #     torch_dtype=torch.float16,  # Load in 16-bit to save RAM/VRAM
    # ).to(device)
    #
    # base_unet.eval()

    base_transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.float16,  # Load in 16-bit to save RAM/VRAM
    ).to(device)

    base_transformer.eval()

    # wrapped_model = SDXLUNetWrapper(base_unet)
    # wrapped_model = SD15UNetWrapper(base_unet)

    wrapped_model = FLux2TransformerWrapper(base_transformer, height=1024, width=1024)

    print("Generating dummy inputs for 1024 resolution...")
    # 3. Create dummy inputs matching SDXL's native architecture
    # Latent for 1024x1024 (1024 / 8 = 128)
    # Latent for 512x512 would be (512 / 8 = 64)
    dummy_latent = torch.rand(1, 4, 128, 128, dtype=torch.float16).to(device)
    dummy_timestep = torch.tensor([10], dtype=torch.float16).to(device)

    # Prompt embeddings
    dummy_prompt_embeds = torch.rand(1, 2560, 40960, dtype=torch.float16).to(device)

    #
    dummy_pooled_text_embeds = torch.rand(1, 1280, dtype=torch.float16).to(device)

    # time_ids represents
    # [original_width, original_height, crop_coords_top, crop_coords_left, target_width, target_height]
    dummy_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float16
    ).to(device)

    # 4. Group inputs exactly as the Wrapper's forward function expects them
    inputs = (
        dummy_latent,
        dummy_timestep,
        dummy_prompt_embeds,
        # dummy_pooled_text_embeds,
        # dummy_time_ids,
    )

    print("Profiling model")
    flops, params = profile(wrapped_model, inputs=inputs, verbose=False)
    print("\n" + "-" * 40)
    print(f"Model: " + model_path.split("/")[-1])
    print(f"Resolution: 1024")
    print(f"FLOPs (1 step) = {clever_format([flops], format='%.2f')}")
    print(f"Params         = {clever_format([params], format='%.2f')}")
