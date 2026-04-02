---
name: cv-evaluate
description: "Evaluate trained CV models with accuracy, confusion matrix, per-class metrics, mAP, mIoU, and Grad-CAM visualizations."
aliases: [eval cv, cv metrics, image model eval, confusion matrix cv, grad-cam]
extends: ml-automation
user_invocable: true
---

# CV Evaluate

Evaluate a trained computer vision model on a test set. Computes task-appropriate metrics (accuracy/F1 for classification, mAP for detection, mIoU for segmentation), generates confusion matrices, per-class breakdowns, and optionally produces Grad-CAM heatmap visualizations for interpretability.

## When to Use

- You have a trained model checkpoint and want to measure performance on a held-out test set
- You need per-class precision/recall/F1 breakdowns or a confusion matrix for error analysis
- You want Grad-CAM visualizations to understand which image regions drive model predictions
- You are comparing multiple models or training runs and need standardized metric reports

## Workflow

1. **Env Check** -- Verify Python environment, GPU availability, and required libraries (PyTorch/TensorFlow, Pillow). Install missing dependencies if needed.
2. **Model Loading** -- Load the trained model from the checkpoint path. Detect task type (classification, detection, segmentation) and number of classes from model architecture or prior reports.
3. **Metric Computation** -- Run inference on the test set and compute task-appropriate metrics: accuracy, precision, recall, F1, confusion matrix (classification); mAP@0.5, mAP@0.75, per-class AP (detection); mIoU, pixel accuracy, per-class IoU (segmentation).
4. **Grad-CAM Visualization** -- If `--grad-cam` is enabled, generate heatmap overlays on a sample of test images showing activation regions. Save visualizations to the output directory.

## Report Bus Integration

Produces `cv_engineer_report.json` (updated) with keys: `evaluation_metrics`, `confusion_matrix`, `per_class_metrics`, `grad_cam_samples`, `test_set_size`. Consumed by `cv-deploy` to document model performance alongside the deployed artifact.

## Full Specification

Usage: `/cv-evaluate <model_path> <dataset_path> [--task classification|detection|segmentation] [--grad-cam]`

Agent: **cv-engineer**

See `commands/cv-evaluate.md` for the complete workflow.
