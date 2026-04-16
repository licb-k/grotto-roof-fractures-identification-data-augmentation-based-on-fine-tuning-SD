import argparse
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from diffuserslocal.src.diffusers.models.Finetuningnet import FinetuningnetModel


logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"暂不支持的 text encoder 类型: {model_class}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FinetuningNet without conditioning image input.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")

    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")    
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)

    parser.add_argument("--validation_prompt", type=str, default=None)

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("必须提供 --dataset_name 或 --train_data_dir 其中一个。")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("--dataset_name 和 --train_data_dir 只能二选一。")

    if args.resolution % 8 != 0:
        raise ValueError("--resolution 必须能被 8 整除。")

    return args


def make_train_dataset(args, tokenizer, accelerator):
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.train_data_dir, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names
    image_column = args.image_column
    caption_column = args.caption_column

    if image_column not in column_names:
        raise ValueError(f"数据集里找不到图像列: {image_column}")
    if caption_column not in column_names:
        raise ValueError(f"数据集里找不到文本列: {caption_column}")

    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def tokenize_captions(examples):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, list):
                captions.append(random.choice(caption))
            else:
                raise ValueError(f"{caption_column} 列必须是字符串或字符串列表。")

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_train(examples):
        images = [Image.open(image).convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]
        examples["pixel_values"] = images
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    logger.info("Initializing FinetuningNet from unet")
    tuningnet = FinetuningnetModel.from_unet(unet)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    tuningnet.train()

    optimizer = torch.optim.AdamW(
        tuningnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, tokenizer, accelerator)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    tuningnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        tuningnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "train_Finetuningnet",
            config={
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "resolution": args.resolution,
                "train_batch_size": args.train_batch_size,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
            },
        )

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.num_train_epochs}")
    logger.info(f"Batch size per device = {args.train_batch_size}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps"
    )

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(tuningnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents.float(), noise.float(), timesteps
                ).to(dtype=weight_dtype)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                down_block_res_samples, mid_block_res_sample = tuningnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[x.to(dtype=weight_dtype) for x in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(tuningnet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_tuningnet = accelerator.unwrap_model(tuningnet)

        if hasattr(unwrapped_tuningnet, "save_pretrained"):
            unwrapped_tuningnet.save_pretrained(args.output_dir)
        else:
            torch.save(unwrapped_tuningnet.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
