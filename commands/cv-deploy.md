# /cv-deploy

Deploy a trained CV model via ONNX export, serving endpoint, or API.

## Usage

```
/cv-deploy <model_path> [--target onnx|torchserve|tfserving|api] [--port 8000] [--quantize fp16|int8]
```

- `model_path`: path to trained model checkpoint (`.pt`, `.pth`, `.h5`)
- `--target`: deployment target (default: onnx)
- `--port`: port for serving endpoints (default: 8000)
- `--quantize`: optional quantization (fp16 or int8)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify model file exists and is loadable
4. Detect model framework (PyTorch or TensorFlow)
5. Report: model framework, architecture, input shape

### Stage 1: Model Export

Based on `--target`:

**ONNX:**
1. Export model to ONNX format with dynamic batch size
2. Validate exported ONNX model (`onnx.checker.check_model`)
3. Optimize ONNX graph (constant folding, redundant node elimination)
4. Apply quantization if `--quantize` specified
5. Benchmark: original vs. ONNX vs. quantized inference latency
6. Save to `models/model.onnx`

**TorchServe:**
1. Create custom handler (`src/handler.py`):
   - `initialize`: load model, set device
   - `preprocess`: decode image, resize, normalize
   - `inference`: forward pass
   - `postprocess`: format predictions (labels, boxes, masks)
2. Generate MAR file with `torch-model-archiver`
3. Generate `config/torchserve_config.properties`
4. Generate Dockerfile for TorchServe

**TensorFlow Serving:**
1. Export as SavedModel with serving signatures
2. Configure input/output tensor specs
3. Generate `config/tf_serving_config.txt` (model versioning, batching)
4. Generate Dockerfile with TF Serving base image

**FastAPI:**
1. Generate `src/app.py`:
   - `/health` endpoint (GET)
   - `/predict` endpoint (POST) -- accepts image upload or base64
   - `/predict/batch` endpoint (POST) -- batch image prediction
   - Request/response Pydantic models
   - Image preprocessing pipeline
   - CORS middleware
2. Generate `src/inference.py`:
   - Model loading and caching
   - Preprocessing (resize, normalize)
   - Postprocessing (class labels, bounding boxes, masks)
3. Generate `requirements.txt` with deployment dependencies

### Stage 2: Docker Configuration

1. Generate `Dockerfile`:
   - Multi-stage build (builder + runtime)
   - CUDA base image if GPU deployment
   - Model file copy
   - Health check
   - Non-root user
2. Generate `docker-compose.yml`:
   - Service configuration
   - Volume mounts for models
   - GPU support (if applicable)
   - Port mapping
3. Generate `.dockerignore`

### Stage 3: Testing

1. Generate `tests/test_inference.py`:
   - Load model and run prediction on sample image
   - Verify output format and shape
   - Benchmark inference latency (mean, p50, p95, p99)
2. Generate `tests/test_api.py` (for API target):
   - Health check test
   - Single image prediction test
   - Batch prediction test
   - Invalid input handling test

### Stage 4: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-deployer", {
    "status": "completed",
    "target": target,
    "model_path": model_path,
    "export_path": export_path,
    "original_size_mb": original_size,
    "exported_size_mb": exported_size,
    "quantization": quantize_mode,
    "latency_ms": {"original": orig_latency, "exported": export_latency},
    "generated_files": generated_files,
    "startup_command": startup_cmd,
    "test_command": test_cmd,
    "recommendations": recommendations
})
```

Write report to `reports/cv_deployment_report.json`.

Print: target, export path, model size comparison, latency benchmarks, startup command.
