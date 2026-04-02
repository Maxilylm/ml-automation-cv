---
name: cv-analyze
description: "Analyze image datasets: class distribution, resolution statistics, quality assessment (blur, duplicates, corrupt), label verification, and sample grid generation."
aliases: [image eda, image analysis, cv dataset analysis, image dataset]
extends: spark
user_invocable: true
---

# CV Analyze

Run comprehensive analysis on an image dataset. Computes class distribution and imbalance metrics, resolution statistics, detects quality issues (blurry, duplicate, corrupt images), verifies annotation labels (COCO, VOC, YOLO, ImageFolder), and generates sample grids per class.

## When to Use

- You have a new image dataset and need to understand its composition before training
- You want to detect class imbalance, resolution inconsistencies, or annotation errors
- You need a visual sample grid to verify labels match the actual images
- You are starting a CV project and need a baseline audit of data quality

## Workflow

1. **Env Check** -- Verify Python environment and required libraries (Pillow, cv_utils). Install missing dependencies if needed.
2. **Dataset Inventory** -- Scan the dataset path for image files and auto-detect annotation format (COCO JSON, VOC XML, YOLO TXT, or ImageFolder structure).
3. **Class Distribution** -- Count instances per class, compute imbalance ratio, and flag severe skew (>5:1 ratio).
4. **Resolution Stats** -- Compute min/max/mean/median/std for width and height across all images. Report channel distribution (RGB, grayscale, RGBA).
5. **Quality Assessment** -- Detect corrupt files (cannot be opened), near-duplicate images (perceptual hashing), and blurry samples (Laplacian variance).
6. **Sample Grid** -- Generate a visual grid of representative images per class, saved to the output directory for quick human review.

## Report Bus Integration

Produces `cv_analyst_report.json` with keys: `total_images`, `classes`, `class_distribution`, `resolution_stats`, `annotation_format`, `issues`, `quality_assessment`. Downstream skills (`cv-preprocess`, `cv-augment`, `cv-train`) consume this report to inherit dataset metadata without re-scanning.

## Full Specification

Usage: `/cv-analyze <dataset_path> [--format coco|voc|yolo|imagefolder] [--output reports/]`

Agent: **cv-analyst**

See `commands/cv-analyze.md` for the complete workflow.
