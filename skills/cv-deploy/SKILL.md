---
name: cv-deploy
description: "Deploy CV models via ONNX export, TorchServe, TensorFlow Serving, or FastAPI endpoints with optional quantization."
aliases: [deploy cv, onnx export, torchserve, tensorflow serving, cv api, model export]
extends: ml-automation
user_invocable: true
---

# CV Deploy

Deploy a trained computer vision model to production. Supports ONNX export with graph optimization and quantization (FP16, INT8), TorchServe with custom handlers, TensorFlow Serving with model versioning, and FastAPI image prediction endpoints. Includes Docker configuration, inference benchmarking, and test generation.

## Full Specification

See `commands/cv-deploy.md` for the complete workflow.
