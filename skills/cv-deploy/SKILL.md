---
name: cv-deploy
description: "Deploy CV models via ONNX export, TorchServe, TensorFlow Serving, or FastAPI endpoints with optional quantization."
aliases: [deploy cv, onnx export, torchserve, tensorflow serving, cv api, model export]
extends: spark
user_invocable: true
---

# CV Deploy

Deploy a trained computer vision model to production. Supports ONNX export with graph optimization and quantization (FP16, INT8), TorchServe with custom handlers, TensorFlow Serving with model versioning, and FastAPI image prediction endpoints. Includes Docker configuration, inference benchmarking, and test generation.

## When to Use

- You have a trained model and need to export it to ONNX for cross-platform inference
- You want to serve predictions via TorchServe, TensorFlow Serving, or a REST API
- You need to quantize a model to FP16 or INT8 for faster inference or edge deployment
- You want a complete deployment package with Docker config, smoke tests, and benchmarks

## Workflow

1. **Env Check** -- Verify Python environment and required export/serving libraries (torch, onnx, onnxruntime, fastapi). Install missing dependencies if needed.
2. **Model Export** -- Export the trained model to the target format (ONNX, TorchScript, SavedModel). Apply graph optimizations and optional quantization (FP16, INT8). Validate the exported model against reference outputs.
3. **Server Setup** -- Generate serving configuration for the chosen target: TorchServe MAR archive and config, TensorFlow Serving model directory, or FastAPI endpoint with image preprocessing. Include Dockerfile and requirements.
4. **Smoke Test** -- Run inference on sample images against the deployed endpoint. Compare outputs with the original model to verify numerical consistency. Report latency benchmarks (p50, p95, p99).

## Report Bus Integration

Produces `cv_deployer_report.json` with keys: `export_format`, `model_size_mb`, `quantization`, `serving_target`, `endpoint_url`, `latency_benchmarks`, `smoke_test_passed`. Consumed by downstream monitoring or CI/CD pipelines.

## Full Specification

Usage: `/cv-deploy <model_path> [--target onnx|torchserve|tfserving|api] [--quantize fp16|int8]`

Agent: **cv-deployer**

See `commands/cv-deploy.md` for the complete workflow.
