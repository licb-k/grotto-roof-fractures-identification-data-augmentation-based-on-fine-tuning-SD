import torch
import numpy as np
from PIL import Image

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from diffuserslocal.src.diffusers.models.Finetuningnet import FinetuningnetModel


# =========================
# Parameter Settings
# =========================
BASE_MODEL_DIR = "runwayml/stable-diffusion-v1-5"
FINETUNINGNET_DIR = "path/to/weights"

PROMPT = "Thermal infrared image of rock surface with crack"
NEGATIVE_PROMPT = ""

HEIGHT = 640
WIDTH = 640
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
SEED = 42
OUTPUT_IMAGE = "finetuningnet_result.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# =========================
# Load Models
# =========================
print("Loading tokenizer / text_encoder / vae / unet / scheduler...")
tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae", torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")

print("Loading Finetuningnet...")
tuningnet = FinetuningnetModel.from_pretrained(
    FINETUNINGNET_DIR,
    torch_dtype=dtype,
).to(device)

text_encoder.eval()
vae.eval()
unet.eval()
tuningnet.eval()


# =========================
# Text Encoding
# =========================
def encode_prompt(prompt_list):
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids)[0]
    return prompt_embeds

prompt_embeds = encode_prompt([PROMPT])
negative_prompt_embeds = encode_prompt([NEGATIVE_PROMPT])

# Classifier-Free Guidance (CFG)
prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)


# =========================
# Initialize Latents
# =========================
generator = torch.Generator(device=device).manual_seed(SEED)

latent_h = HEIGHT // 8
latent_w = WIDTH // 8

latents = torch.randn(
    (1, unet.config.in_channels, latent_h, latent_w),
    generator=generator,
    device=device,
    dtype=dtype,
)

scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
latents = latents * scheduler.init_noise_sigma


# =========================
# Denoising Inference
# =========================
print("Start inference...")
with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)  # unconditional + conditional
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Pass through Finetuningnet first
        down_block_res_samples, mid_block_res_sample = tuningnet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )

        # Then pass through the original SD1.5 UNet
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # Apply CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        # Update latents using the scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0 or (i + 1) == len(scheduler.timesteps):
            print(f"Step {i + 1}/{len(scheduler.timesteps)}")

    # Decode latents into image
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]

# Post-processing
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image[0] * 255).astype(np.uint8)

img = Image.fromarray(image)
img.save(OUTPUT_IMAGE)
print(f"Saved image to: {OUTPUT_IMAGE}")