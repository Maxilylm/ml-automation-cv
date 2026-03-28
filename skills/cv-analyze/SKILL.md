---
name: cv-analyze
description: "Analyze image datasets: class distribution, resolution statistics, quality assessment (blur, duplicates, corrupt), label verification, and sample grid generation."
aliases: [image eda, image analysis, cv dataset analysis, image dataset]
extends: ml-automation
user_invocable: true
---

# CV Analyze

Run comprehensive analysis on an image dataset. Computes class distribution and imbalance metrics, resolution statistics, detects quality issues (blurry, duplicate, corrupt images), verifies annotation labels (COCO, VOC, YOLO), and generates sample grids per class.

## Full Specification

See `commands/cv-analyze.md` for the complete workflow.
