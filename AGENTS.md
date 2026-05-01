# spark-cv — Cortex Code Extension

Computer vision automation. Image preprocessing, CNN training, object detection, image classification, data augmentation, and model deployment. Requires spark-core installed.

## Available Agents

| Agent | When to use |
|---|---|
| `cv-analyst` | User wants to explore an image dataset, check class distributions, visualize samples, or profile data quality |
| `cv-engineer` | User wants to train a CNN, set up object detection, or implement an image classification pipeline |
| `cv-deployer` | User wants to serve a vision model as an API, containerize it, or deploy to a cloud endpoint |

## Available Skills

| Skill | Trigger |
|---|---|
| `/cv-preprocess` | "preprocess images", "resize and normalize images", "build image preprocessing pipeline" |
| `/cv-analyze` | "analyze image dataset", "image EDA", "check class balance", "visualize image samples" |
| `/cv-augment` | "augment images", "data augmentation", "add rotations/flips/crops", "expand training set" |
| `/cv-train` | "train a CNN", "image classification model", "object detection training", "fine-tune ResNet" |
| `/cv-evaluate` | "evaluate vision model", "confusion matrix for images", "precision/recall per class" |
| `/cv-deploy` | "deploy vision model", "serve image classifier", "create inference API for images" |

## Routing

- Dataset exploration, class distribution → `cv-analyst`
- Training, detection, classification → `cv-engineer`
- Serving, deployment, containerization → `cv-deployer`
- Fallback → spark-core orchestrator
