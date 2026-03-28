---
name: cv-augment
description: "Configure and preview data augmentation strategies (light, medium, heavy, custom) with annotation-aware transforms for CV training."
aliases: [image augmentation, data augmentation, augment images, cv augment, cutmix, mixup]
extends: ml-automation
user_invocable: true
---

# CV Augment

Configure and preview data augmentation pipelines for computer vision training. Offers preset strategies (light, medium, heavy) and custom transform selection. Generates side-by-side preview grids showing original and augmented images. Supports annotation-aware transforms that correctly update bounding boxes and segmentation masks.

## Full Specification

See `commands/cv-augment.md` for the complete workflow.
