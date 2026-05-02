# spark-cv

Computer vision extension for [ml-automation](https://github.com/BLEND360/ml-automation-core).

## Prerequisites

- [ml-automation](https://github.com/BLEND360/ml-automation-core) core plugin (>= v1.8.0)
- Claude Code CLI
- PyTorch/TensorFlow for training and deployment commands
- Pillow for image analysis

## Installation

```bash
claude plugin add /path/to/spark-cv
```

## What's Included

### Agents

| Agent | Purpose | Hooks Into |
|---|---|---|
| `cv-analyst` | Image dataset analysis, class distribution, quality assessment, label verification | `after-eda` |
| `cv-engineer` | CV model training: classification, detection, segmentation with transfer learning | `after-training` |
| `cv-deployer` | CV model deployment: ONNX export, TorchServe, TensorFlow Serving, API endpoints | *(direct invocation)* |

### Commands

| Command | Purpose |
|---|---|
| `/cv-analyze` | Image dataset EDA (class balance, resolution, quality, sample grid) |
| `/cv-preprocess` | Image preprocessing pipeline (resize, normalize, augment, split) |
| `/cv-train` | Train CV model (classification, detection, segmentation) |
| `/cv-evaluate` | Evaluate CV model (accuracy, confusion matrix, Grad-CAM) |
| `/cv-augment` | Configure and preview data augmentation strategies |
| `/cv-deploy` | Deploy CV model (ONNX export, serving endpoint, API) |

## Getting Started

```bash
# Analyze an image dataset
/cv-analyze ./data/images/ --format imagefolder

# Preprocess images for training
/cv-preprocess ./data/images/ --target-size 224x224 --split 0.7,0.15,0.15

# Train a classification model
/cv-train ./processed/ --task classification --backbone resnet50 --epochs 50

# Evaluate the trained model
/cv-evaluate ./models/best.pt ./processed/test/ --grad-cam

# Configure augmentation
/cv-augment ./data/images/ --strategy heavy --preview 10

# Deploy as ONNX
/cv-deploy ./models/best.pt --target onnx --quantize fp16
```

## How It Integrates

When installed alongside the core plugin:

1. **Automatic routing** -- Tasks mentioning image analysis, CNN training, object detection, or CV deployment are routed to CV agents
2. **Core workflow hooks** -- When running `/team-coldstart`:
   - `cv-analyst` fires at `after-eda` to detect and analyze image datasets
   - `cv-engineer` fires at `after-training` to add CV-specific training metrics
3. **Core agent reuse** -- Commands use eda-analyst, developer, ml-theory-advisor from core

## License

MIT
