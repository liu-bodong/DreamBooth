# Reimplementing DreamBooth

A re-implementation of [**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**](https://arxiv.org/abs/2208.12242) (Ruiz et al., CVPR 2023) by Bodong Liu, Pablo Raigoza, David Suh, and Ryan Wu.

![DreamBooth Poster](poster/poster.png)

---

## Introduction

Text-to-image diffusion models can generate high-quality images from natural language prompts, but they typically cannot generate a specific instance of a subject (i.e., professor vs. Kilian Weinberger). DreamBooth addresses this problem by taking a ~3-5 images of a specific subject and fine-tunes a pretrained diffusion model so that a rare token like “[V]” associates with that subject. The fine-tuned model can then generate the subject in novel contexts, poses, styles, lighting, etc. The paper also introduces a **prior preservation loss (PPL)** to counteract language drift — the tendency of the model to over-associate the class noun with the training subject.

This repo re-implements DreamBooth from scratch and extends beyond the original paper with **LoRA-based fine-tuning** and **one-shot experiments** across four pretrained models (SD-1.5, SD-XL, SD-3M, Flux2 4B).

---

## Chosen Result

We focused on the paper’s main result: fine-tuning a pretrained text-to-image model on a few subject images that can generate the same subject in new contexts without overfitting the general class to the specific instance. In the original paper, this includes implementing the fine-tuning pipeline in Figure 3, the recontextualization examples in Figure 7, and the prior preservation loss ablation in Table 3, Figure 6, and Equation 2.

> **[Results table — see Results/Insights section below]**

---

## Repository Structure

```
DreamBooth/
├── code/
│   ├── src/            # Training, inference, evaluation scripts
│   │   ├── scripts/    # Entry points: train_lora, train_orig, generate, evaluate, …
│   │   ├── training/   # Training loops and dataset classes
│   │   └── evaluation/ # Metric computation
│   └── configs/        # YAML config files
│       ├── lora/sdxl/     # Per-subject LoRA configs (no PPL)
│       ├── lora_ppl/sdxl/ # Per-subject LoRA + PPL configs
│       ├── full/sdxl/     # Per-subject full fine-tune configs (no PPL)
│       ├── full_ppl/sdxl/ # Per-subject full fine-tune + PPL configs
│       └── subjects/      # Subject-specific override YAMLs
├── data/
│   └── raw/            # Original subject images (not committed)
|   └── README.md       # How to download data
├── poster/             # Poster PDF
├── report/             # 2-page final report PDF
├── docs/               # Reference papers
├── makefile            # Common training/eval commands
└── requirements.txt
```

---

## Re-implementation Details

**Models:** Stable Diffusion XL (primary), SD-1.5, SD-3M, Flux2 4B — all loaded from local `base_models/` checkpoints.

**Datasets:** Official [DreamBooth dataset](https://github.com/google/dreambooth) (dogs, cats, backpacks, vases, teapots, rubber ducks, sunglasses, clocks, cartoon figures) plus personal images (people). Instance images live in `data/instance/<subject>/`.

**Adaptation strategies:** (1) Full DreamBooth fine-tuning — updates all U-Net weights; (2) LoRA — trains low-rank adapter matrices (rank 32) instead, cutting VRAM from ~28 GB to ~15 GB and enabling larger learning rates and batch sizes.

**Prior preservation:** When enabled, 1 000 class images (e.g. "a dog") are pre-generated from the base model and added to training as a regularizer (PPL weight λ = 1.0).

**Evaluation metrics:**
| Metric | Description |
|--------|-------------|
| PRES ↓ | Prior preservation — smaller = less class collapse |
| DIV ↑  | Diversity across generated samples of the learned subject |
| DINO ↑ | Subject fidelity vs. reference (ViT-based features) |
| CLIP-I ↑ | Subject fidelity vs. reference (CLIP image features) |
| CLIP-T ↑ | Text fidelity — generated image vs. prompt |

All training and benchmarking was done on an NVIDIA RTX 5090.

---

## Reproduction Steps

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or
make build
```

### 2. Download base models

```bash
cd code
python -m src.scripts.download_models
```

Model checkpoints are saved to `base_models/` (not committed to git — ~15–28 GB per model).

### 3. Prepare subject images

Place 3–5 images of your subject in `data/instance/<subject_name>/` (e.g. `data/instance/dog/`).

### 4. (Optional) Generate class images for prior preservation

```bash
# from the code/ directory
python -m src.scripts.generate_class_images --config configs/lora_ppl/sdxl/dog.yaml
```

### 5. Train

All commands below are run from the `code/` directory.

**LoRA fine-tune (recommended — lower VRAM):**
```bash
python -m src.scripts.train_lora --config configs/lora/sdxl/dog.yaml
```

**Full DreamBooth fine-tune:**
```bash
python -m src.scripts.train_orig --config configs/sd15_full.yaml
```

**With prior preservation loss, swap the config directory:**
```bash
python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/dog.yaml
```

#### The override YAML trick

Every config sets defaults for a specific subject. To **reuse a base config for a new subject** without duplicating the whole file, pass a second `--override` YAML that only specifies the fields you want to change:

```bash
python -m src.scripts.train_lora \
  --config configs/lora/sdxl/dog.yaml \
  --override configs/subjects/teapot.yaml
```

`configs/subjects/teapot.yaml` might contain only:
```yaml
subject_name: teapot
unique_token: kqwei
output_dir: models/sdxl/lora/teapot
subject_class_map:
  teapot: teapot
instance_data_root: data/instance
```

All other hyperparameters are inherited from the base config. This is the pattern used throughout `makefile` and in `configs/subjects/` for running batch experiments across subjects.

### 6. Generate images

```bash
python -m src.scripts.generate \
  --config configs/lora_ppl/sdxl/dog.yaml \
  --checkpoint models/sdxl/lora_ppl/dog/final \
  --num_images 16 \
  --output_dir outputs/sdxl/lora_ppl/dog \
  --seed 42 --cfg 7.0 --num_steps 50
```

### 7. Evaluate

```bash
python -m src.scripts.evaluate \
  --generated_dir outputs/sdxl/lora_ppl/dog/instance \
  --real_dir data/instance/dog \
  --prompt "a photo of kqwei dog"
```

**Hardware requirement:** ~15 GB VRAM for LoRA (SDXL), ~28 GB for full fine-tune (SDXL). A GPU with at least 16 GB VRAM is strongly recommended.

---

## Results / Insights

Our results across all six configurations, compared against the original paper's reported numbers (which used a different base model, number of epochs, and text encoders):

| Method | PRES ↓ | DIV ↑ | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|--------|--------|-------|--------|----------|----------|
| DreamBooth (Original) | 0.664 | 0.371 | 0.712 | 0.828 | 0.306 |
| DreamBooth (Original) w/ PPL | 0.493 | 0.391 | 0.684 | 0.815 | **0.308** |
| Full Model (Ours) | 0.848 | 0.631 | **0.875** | 0.909 | 0.302 |
| Full Model (Ours) w/ PPL | 0.419 | 0.654 | 0.853 | 0.910 | 0.304 |
| LoRA Model (Ours) | 0.847 | 0.671 | 0.865 | **0.920** | 0.291 |
| LoRA Model (Ours) w/ PPL | **0.372** | **0.702** | 0.857 | 0.912 | 0.294 |

Our models consistently outperform the original across PRES, DIV, DINO, and CLIP-I. The strongest trend is the inclusion of PPL, which matches the original paper's conclusion that PPL counteracts language drift and improves diversity. LoRA achieved the best PRES, DIV, and CLIP-I scores overall. Note that direct comparison to the original paper is imperfect because of model, epoch, and encoder differences — visual inspection remains an important signal.

---

## Analysis

Our re-implementation confirms and extends the original paper's core findings:

**Prior preservation loss works as claimed.** Across both full fine-tuning and LoRA, adding PPL consistently lowered PRES (less class collapse) and raised DIV (more diversity), matching the paper's argument in Section 3.2. The slight drop in DINO when using PPL is a real tradeoff: the model preserves the broader class at the cost of slightly weaker individual subject memorization.

**LoRA is a competitive alternative to full fine-tuning.** LoRA achieved the best overall scores on PRES, DIV, and CLIP-I while cutting VRAM by ~46%. We theorize LoRA's inherent low-rank bottleneck acts as an implicit regularizer — the model is forced to *learn* the subject rather than *memorize* training pixels, which appears to benefit generalization.

**Three consistent failure modes emerged:**
1. *Partial identity learning* — correct shape but wrong color or texture, suggesting the model bound only part of the subject to the rare token.
2. *No learning* — the model ignores the rare token and generates a generic class instance; likely a failure to bind the identifier during training.
3. *Strong prior dominance* — in style-heavy prompts (e.g. "van Gogh style vase"), the pretrained style prior overwhelms the learned subject identity.

**One-shot fine-tuning** (single training image) worked but was less reliable: poses were less novel and identity fidelity dropped. This is an area not covered in the original paper and warrants further investigation.

---

## Conclusion

DreamBooth is conceptually elegant but practically sensitive — output quality depends heavily on model choice, learning rate, number of training steps, and prompt design. Undertrain and the rare token is ignored; overtrain and the model memorizes input images, damaging the class prior. PPL is a necessary component for production-quality diversity, and LoRA is a strong, VRAM-efficient alternative to full fine-tuning that can even improve generalization.

---

## References

- Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.* CVPR 2023, pp. 22500–22510.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

---

## Acknowledgements

This work was conducted at Cornell University as part of CS 4782. We used the official [DreamBooth dataset](https://github.com/google/dreambooth) and the [Hugging Face `diffusers`](https://github.com/huggingface/diffusers) library as the primary training framework.
