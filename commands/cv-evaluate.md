# /cv-evaluate

Evaluate a trained CV model with accuracy, confusion matrix, per-class metrics, and Grad-CAM visualizations.

## Usage

```
/cv-evaluate <model_path> <dataset_path> [--task classification|detection|segmentation] [--grad-cam] [--output reports/]
```

- `model_path`: path to trained model checkpoint (`.pt`, `.pth`, `.h5`, `.onnx`)
- `dataset_path`: test/validation dataset directory
- `--task`: CV task type (default: auto-detect)
- `--grad-cam`: generate Grad-CAM visualizations for misclassified samples
- `--output`: output directory for reports (default: `reports/`)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify model file exists and is loadable
4. Verify dataset path exists and contains images
5. Auto-detect task type from model architecture if not specified

### Stage 1: Model Loading

1. Load model checkpoint (PyTorch, TensorFlow, or ONNX)
2. Detect architecture and task type from model structure
3. Set model to evaluation mode
4. Report: model architecture, parameter count, input size

### Stage 2: Inference

1. Create test DataLoader with appropriate transforms (no augmentation)
2. Run inference on all test samples
3. Collect predictions, ground truth labels, and confidence scores
4. Report: total samples, inference time, throughput (images/sec)

### Stage 3: Classification Metrics (if classification task)

1. **Overall accuracy** -- top-1 and top-5 accuracy
2. **Per-class metrics** -- precision, recall, F1 for each class
3. **Confusion matrix** -- NxN matrix with counts and percentages
4. **Confidence analysis** -- reliability diagram, ECE (Expected Calibration Error)
5. **Error analysis** -- most confused class pairs, hardest samples

### Stage 4: Detection Metrics (if detection task)

1. **mAP** -- mean Average Precision at IoU thresholds (0.5, 0.75, 0.5:0.95)
2. **Per-class AP** -- Average Precision for each object class
3. **Precision-Recall curves** -- per class and overall
4. **Size analysis** -- AP for small, medium, large objects
5. **Error breakdown** -- classification errors vs. localization errors

### Stage 5: Segmentation Metrics (if segmentation task)

1. **mIoU** -- mean Intersection over Union across classes
2. **Per-class IoU** -- IoU for each semantic class
3. **Pixel accuracy** -- overall and per-class
4. **Boundary F1** -- precision at object boundaries
5. **Confusion matrix** -- pixel-level class confusion

### Stage 6: Grad-CAM Visualization (if --grad-cam)

1. Select target layer (last convolutional layer by default)
2. Generate Grad-CAM heatmaps for:
   - Correctly classified samples (top confidence)
   - Misclassified samples (all or top-N)
   - Edge cases (low confidence correct predictions)
3. Overlay heatmaps on original images
4. Save visualization grid to `reports/grad_cam/`

### Stage 7: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-engineer", {
    "status": "completed",
    "model_path": model_path,
    "task": task,
    "total_samples": total_samples,
    "metrics": metrics,
    "confusion_matrix": confusion_matrix,
    "per_class_metrics": per_class_metrics,
    "inference_throughput": throughput,
    "grad_cam_generated": grad_cam_flag,
    "recommendations": recommendations
})
```

Write report to `reports/cv_evaluation_report.json`.

Print summary table: overall metric, per-class breakdown, worst performing classes, recommendations.
