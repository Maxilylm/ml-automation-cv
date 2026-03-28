---
name: cv-preprocess
description: "Image preprocessing pipeline: resize, normalize, augment, and split into train/val/test sets with stratified sampling."
aliases: [image preprocess, image resize, image normalize, cv pipeline, image split]
extends: ml-automation
user_invocable: true
---

# CV Preprocess

Build an image preprocessing pipeline. Resizes images to a target resolution, applies normalization (ImageNet, min-max, or per-dataset standardization), and creates stratified train/val/test splits preserving class ratios. Updates annotations (bounding boxes, masks) to match resized images.

## Full Specification

See `commands/cv-preprocess.md` for the complete workflow.
