# DreamBooth

An implementation of [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) (Ruiz et al., Google Research 2022).

The paper PDF is in `docs/DreamBooth_Paper.pdf`.

## Repository Layout

```
DreamBooth/
├── src/            # Core logic
├── tests/          # All testing code
│   ├── unit/       # Testing individual functions
│   └── integration/# Testing the full pipeline
├── data/           # Core logic
│   ├── raw/        # Original subject images
│   ├── interim/    # Intermediate preprocessed files
│   └── processed/  # Final datasets ready for training
├── models/         # Saved checkpoints (not committed — see below)
├── notebooks/      # Exploration and demo notebooks only
├── reports/        # Generated samples, metrics, and experiment write-ups
├── docs/           # Reference papers and documentation
├── configs/        # Training config files (hyperparameters, paths)
└── requirements.txt # Python dependencies
```

**Rules for contributors:**
- All training/inference code goes in `src/`, not notebooks
- Large files (model weights, image datasets) must not be committed — document how to obtain or generate them
- Each experiment should have a corresponding config file in `configs/`

## Dataset

The DreamBooth team released their dataset: (https://arxiv.org/abs/2208.12242).

---

## Cheatsheet

### Subject setup
Put instance images in `data/processed/instance/<subject_name>/`.

Create a subject override yaml (e.g. `configs/subjects/xiaobai.yaml`):
```yaml
subject_name: xiaobai
unique_token: kqwei
output_dir: models/xiaobai_lora
with_prior_preservation: false
```

### Train
```bash
# LoRA
python -m src.scripts.train_lora --config configs/dreambooth_lora_sd15.yaml --override configs/subjects/xiaobai.yaml

# Full
python -m src.scripts.train_orig --config configs/dreambooth_full_sd15.yaml --override configs/subjects/xiaobai.yaml
```

### Generate class images (prior preservation only)
```bash
python -m src.scripts.generate_class_images --config configs/dreambooth_lora_sd15.yaml
```

### Generate images
```bash
python -m src.scripts.generate --config configs/dreambooth_lora_sd15.yaml --checkpoint models/xiaobai_lora/final --num_images 8 --output_dir outputs/xiaobai --seed 42
```

### Evaluate
```bash
python -m src.scripts.evaluate --generated_dir outputs/xiaobai --real_dir data/processed/instance/xiaobai --prompt "a photo of kqwei dog"
```

### Tips
- Use `--checkpoint models/.../checkpoint-XXXXXX` to evaluate an intermediate checkpoint
- Set `with_prior_preservation: false` in override to skip class image generation
- Set `center_crop: true` if instance images are not already square
