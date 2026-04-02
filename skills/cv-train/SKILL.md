---
name: cv-train
description: "Train CV models for classification, detection, or segmentation with transfer learning from pretrained backbones (ResNet, EfficientNet, YOLO, U-Net)."
aliases: [train cv, image classification, object detection train, segmentation train, transfer learning cv]
extends: ml-automation
user_invocable: true
---

# CV Train

Train a computer vision model with transfer learning. Supports image classification (ResNet, EfficientNet, ViT), object detection (YOLO, Faster R-CNN), and semantic/instance segmentation (U-Net, DeepLabV3+, Mask R-CNN). Configures optimizer, scheduler, mixed precision, early stopping, and checkpoint management.

## When to Use

- You have a preprocessed image dataset and want to train a classification, detection, or segmentation model
- You want to leverage pretrained backbones via transfer learning to reduce training time and improve accuracy
- You need end-to-end training with automatic optimizer, scheduler, and early stopping configuration
- You want reproducible training runs with checkpointing and metric logging

## Workflow

1. **Env Check** -- Verify Python environment, GPU availability (CUDA/MPS), and required frameworks (PyTorch or TensorFlow). Install missing dependencies if needed.
2. **Data Loading** -- Create DataLoaders from the preprocessed dataset splits. Apply normalization and augmentation pipelines from prior reports if available.
3. **Model Setup (Transfer Learning)** -- Load the pretrained backbone, replace the head for the target number of classes, and freeze/unfreeze layers according to the fine-tuning strategy. Configure optimizer, learning rate scheduler, and loss function.
4. **Training** -- Run the training loop with mixed precision, gradient clipping, early stopping, and periodic checkpointing. Log metrics (loss, accuracy/mAP/mIoU) per epoch.
5. **Evaluation** -- Evaluate the best checkpoint on the validation set. Produce a summary of final metrics and save the trained model.

## Report Bus Integration

Produces `cv_engineer_report.json` with keys: `task`, `backbone`, `framework`, `epochs_completed`, `best_metrics`, `checkpoint_path`, `training_history`. Consumed by `cv-evaluate` for detailed evaluation and `cv-deploy` for model export.

## Full Specification

Usage: `/cv-train <dataset_path> [--task classification|detection|segmentation] [--backbone resnet50] [--epochs 50] [--framework pytorch|tensorflow]`

Agent: **cv-engineer**

See `commands/cv-train.md` for the complete workflow.
