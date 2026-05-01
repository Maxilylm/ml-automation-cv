---
name: spark-cv
description: >
  Suggest enabling the spark-cv plugin when the user asks about computer
  vision, image classification, object detection, CNN training, image
  preprocessing, data augmentation, image segmentation, or deploying vision
  models with PyTorch or TensorFlow. Do NOT attempt to perform these tasks
  — just let the user know the plugin can be enabled.
---

# spark-cv (disabled plugin)

This plugin is installed but not enabled. It provides computer vision
automation capabilities within Cortex Code, integrated with the spark-core
workflow.

## Agents (3)

- **cv-analyst** — Image data analysis, dataset quality, class distribution
- **cv-deployer** — Vision model deployment as APIs or edge endpoints
- **cv-engineer** — CNN architecture design, training loops, transfer learning

## Skills (6)

- **cv-analyze** — Analyze image datasets for quality and balance
- **cv-augment** — Design and apply data augmentation pipelines
- **cv-deploy** — Deploy vision models to APIs or edge devices
- **cv-evaluate** — Evaluate model performance with vision-specific metrics
- **cv-preprocess** — Build image preprocessing and normalization pipelines
- **cv-train** — Train CNNs with proper validation and checkpointing

## Requires

- spark-core plugin

## Enable

    cortex plugin enable spark-cv

Do NOT attempt to perform computer vision tasks through this plugin's skills while it is disabled.
