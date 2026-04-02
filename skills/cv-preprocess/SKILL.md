---
name: cv-preprocess
description: "Image preprocessing pipeline: resize, normalize, augment, and split into train/val/test sets with stratified sampling."
aliases: [image preprocess, image resize, image normalize, cv pipeline, image split]
extends: spark
user_invocable: true
---

# CV Preprocess

Build an image preprocessing pipeline. Resizes images to a target resolution, applies normalization (ImageNet, min-max, or per-dataset standardization), and creates stratified train/val/test splits preserving class ratios. Updates annotations (bounding boxes, masks) to match resized images.

## When to Use

- You need to resize a heterogeneous-resolution dataset to a uniform size before training
- You want stratified train/val/test splits that preserve class proportions
- You need to apply ImageNet or custom normalization consistently across all splits
- Your annotations (bounding boxes, segmentation masks) must be updated to reflect resized coordinates

## Workflow

1. **Env Check** -- Verify Python environment and required libraries (Pillow, cv_utils). Install missing dependencies if needed.
2. **Dataset Analysis** -- Load dataset metadata (from `cv_analyst_report.json` if available, or scan directly). Detect annotation format and class distribution.
3. **Resize / Normalize** -- Resize all images to the target size using high-quality interpolation. Apply the chosen normalization scheme. Rescale annotation coordinates proportionally.
4. **Stratified Split** -- Split images into train/val/test sets using the specified ratios while maintaining class distribution. Copy images and annotations into the split directory structure.

## Report Bus Integration

Produces `cv_analyst_report.json` (updated) with keys: `split_sizes`, `target_size`, `normalization`, `split_ratios`, `output_path`. Downstream skills (`cv-augment`, `cv-train`) read this report to locate the preprocessed splits.

## Full Specification

Usage: `/cv-preprocess <dataset_path> [--target-size 224x224] [--split 0.7,0.15,0.15] [--normalize imagenet]`

Agent: **cv-analyst**

See `commands/cv-preprocess.md` for the complete workflow.
