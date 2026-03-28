# /cv-augment

Configure and preview data augmentation strategies for CV training.

## Usage

```
/cv-augment <dataset_path> [--strategy light|medium|heavy|custom] [--preview 10] [--output augmented/]
```

- `dataset_path`: image dataset directory
- `--strategy`: augmentation intensity (default: medium)
- `--preview`: number of sample augmentations to generate per image (default: 10)
- `--output`: output directory for augmented images (default: preview only)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify dataset path exists and contains images
4. Detect annotation format for transform-compatible augmentation

### Stage 1: Dataset Analysis

1. Sample images from dataset (10-20 representative images)
2. Compute image statistics: resolution, color distribution, edge density
3. Detect dataset characteristics that inform augmentation strategy:
   - Fine-grained classification (high intra-class similarity) -> more spatial transforms
   - Medical/satellite imagery -> conservative color transforms
   - Object detection -> bbox-safe transforms only
4. Report: dataset characteristics, recommended strategy

### Stage 2: Strategy Configuration

Build augmentation pipeline based on `--strategy`:

**Light** (validation-safe):
- Random horizontal flip (p=0.5)
- Small rotation (+-10 degrees)
- Slight brightness/contrast jitter (+-0.1)

**Medium** (standard training):
- Random horizontal flip (p=0.5)
- Random rotation (+-15 degrees)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- Random resized crop (scale=0.8-1.0)
- Random erasing (p=0.1)

**Heavy** (data-scarce scenarios):
- Random horizontal and vertical flip
- Random rotation (+-30 degrees)
- Strong color jitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
- Random affine (translate=0.1, scale=0.8-1.2, shear=10)
- Gaussian blur (p=0.2)
- Random erasing (p=0.2)
- Cutout / CutMix / MixUp

**Custom**:
- Interactive selection from available transforms
- Parameter tuning per transform
- Save configuration to `config/augmentation_config.json`

### Stage 3: Preview Generation

1. Apply augmentation pipeline to sample images
2. Generate `--preview` augmented versions of each sample
3. Create side-by-side comparison grid (original + augmentations)
4. If annotations: verify bounding boxes / masks are correctly transformed
5. Save preview images to `reports/augmentation_preview/`

### Stage 4: Offline Augmentation (if --output specified)

1. Apply augmentation pipeline to entire dataset
2. Generate augmented copies (configurable multiplier)
3. Update annotations for augmented images
4. Save augmented dataset to `--output` directory
5. Report: original count, augmented count, augmentation ratio

### Stage 5: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-analyst", {
    "status": "completed",
    "strategy": strategy,
    "transforms": transform_list,
    "preview_generated": preview_count,
    "offline_augmented": augmented_count,
    "output_path": output_path,
    "recommendations": recommendations
})
```

Write augmentation config to `config/augmentation_config.json`.

Print: strategy name, transform pipeline, preview location, recommendations.
