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
