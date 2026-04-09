#Fine-tuning Stable Diffusion model (Finetuningnet)

Finetuningnet is a fine-tuning framework developed for training, inference, and evaluation on specific datasets.  
The file structure and part of the code are modified from the [diffusers](https://github.com/huggingface/diffusers/tree/main) project, with additional adjustments and extensions made to meet the requirements of this project.

---

## Overview

This project is mainly used for:

- Fine-tuning model training with Finetuningnet
- Running inference with the fine-tuned model
- Calculating MMD-related metrics for evaluation

---

## Project Structure

The main files in this project are:

- `data_transpath.py` — used to modify dataset paths in annotation files
- `base_model_path.txt` — instructions for placing the base model
- `train_setting.txt` — training parameter settings and launch commands
- `reasoning_net.py` — inference script for the fine-tuned model
- `MMDcul.py` — script for MMD coefficient calculation
- `requirements.txt` — required Python dependencies

---

## Installation

Install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt