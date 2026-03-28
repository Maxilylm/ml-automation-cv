---
name: cv-train
description: "Train CV models for classification, detection, or segmentation with transfer learning from pretrained backbones (ResNet, EfficientNet, YOLO, U-Net)."
aliases: [train cv, image classification, object detection train, segmentation train, transfer learning cv]
extends: ml-automation
user_invocable: true
---

# CV Train

Train a computer vision model with transfer learning. Supports image classification (ResNet, EfficientNet, ViT), object detection (YOLO, Faster R-CNN), and semantic/instance segmentation (U-Net, DeepLabV3+, Mask R-CNN). Configures optimizer, scheduler, mixed precision, early stopping, and checkpoint management.

## Full Specification

See `commands/cv-train.md` for the complete workflow.
