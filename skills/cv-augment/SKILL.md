---
name: cv-augment
description: "Configure and preview data augmentation strategies (light, medium, heavy, custom) with annotation-aware transforms for CV training."
aliases: [image augmentation, data augmentation, augment images, cv augment, cutmix, mixup]
extends: ml-automation
user_invocable: true
---

# CV Augment

Configure and preview data augmentation pipelines for computer vision training. Offers preset strategies (light, medium, heavy) and custom transform selection. Generates side-by-side preview grids showing original and augmented images. Supports annotation-aware transforms that correctly update bounding boxes and segmentation masks.

## When to Use

- Your dataset is small and you need to increase effective training-set size
- You want to compare augmentation intensities before committing to a strategy
- You need annotation-aware transforms that update bounding boxes or masks alongside images
- You are experimenting with CutMix, MixUp, or other advanced augmentation techniques

## Workflow

1. **Env Check** -- Verify Python environment and required libraries (Pillow, torchvision or albumentations). Install missing dependencies if needed.
2. **Dataset Analysis** -- Load dataset metadata (from `cv_analyst_report.json` if available, or scan directly). Determine annotation format and task type to select bbox-safe transforms.
3. **Strategy Selection** -- Choose from preset strategies (light, medium, heavy) or define a custom transform list. For detection/segmentation tasks, automatically restrict to annotation-safe transforms.
4. **Preview** -- Generate a side-by-side grid of original vs. augmented samples for visual review. Saves preview images to the output directory.
5. **Apply** -- Apply the selected augmentation pipeline to the training split, generating augmented copies alongside originals.

## Report Bus Integration

Produces `cv_analyst_report.json` (updated) with keys: `augmentation_strategy`, `transforms`, `preview_path`, `augmented_count`. Consumed by `cv-train` to reproduce the exact augmentation pipeline during training.

## Full Specification

Usage: `/cv-augment <dataset_path> [--strategy light|medium|heavy|custom] [--preview 10]`

Agent: **cv-analyst**

See `commands/cv-augment.md` for the complete workflow.
