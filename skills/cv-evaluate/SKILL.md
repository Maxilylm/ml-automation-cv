---
name: cv-evaluate
description: "Evaluate trained CV models with accuracy, confusion matrix, per-class metrics, mAP, mIoU, and Grad-CAM visualizations."
aliases: [eval cv, cv metrics, image model eval, confusion matrix cv, grad-cam]
extends: ml-automation
user_invocable: true
---

# CV Evaluate

Evaluate a trained computer vision model on a test set. Computes task-appropriate metrics (accuracy/F1 for classification, mAP for detection, mIoU for segmentation), generates confusion matrices, per-class breakdowns, and optionally produces Grad-CAM heatmap visualizations for interpretability.

## Full Specification

See `commands/cv-evaluate.md` for the complete workflow.
