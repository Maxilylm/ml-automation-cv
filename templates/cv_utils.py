"""
CV utilities for the spark-cv extension plugin.

Requires ml_utils.py from the spark core plugin to be present
in the same directory (copied via Stage 0 of CV commands).
"""

import json
import re
from pathlib import Path
from collections import Counter


# --- Relevance Detection ---

CV_INDICATORS = {
    "torchvision",
    "tensorflow",
    "cv2",
    "PIL",
    "Pillow",
    "albumentations",
    "detectron2",
    "ultralytics",
    "mmdet",
    "mmcv",
    "timm",
    "kornia",
    "imgaug",
    "skimage",
}

CV_MODEL_PATTERNS = [
    r"resnet",
    r"efficientnet",
    r"yolo",
    r"vgg",
    r"inception",
    r"mobilenet",
    r"densenet",
    r"convnext",
    r"swin",
    r"vit",
    r"detr",
    r"mask.?rcnn",
    r"faster.?rcnn",
    r"unet",
    r"deeplab",
    r"segformer",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

ANNOTATION_FORMATS = {
    "coco": ["*.json"],
    "voc": ["*.xml"],
    "yolo": ["*.txt"],
}


def detect_cv_relevance(project_path="."):
    """Check if project has computer vision indicators for relevance gating.

    Checks: CV library imports, image directories, annotation files,
    model files, CV model name references.

    Args:
        project_path: root directory of the project

    Returns:
        dict with 'is_cv': bool, 'indicators': list of found indicators
    """
    indicators = []
    project = Path(project_path)

    # Check for image directories
    image_count = 0
    for ext in IMAGE_EXTENSIONS:
        found = list(project.glob(f"**/*{ext}"))[:100]  # limit scan
        image_count += len(found)
    if image_count > 0:
        indicators.append(f"{image_count}+ image files found")

    # Check for annotation files
    for fmt, patterns in ANNOTATION_FORMATS.items():
        for pattern in patterns:
            # COCO JSON: check for 'images' and 'annotations' keys
            if fmt == "coco":
                json_files = list(project.glob(f"**/{pattern}"))[:10]
                for jf in json_files:
                    try:
                        with open(jf) as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "images" in data and "annotations" in data:
                                indicators.append(f"COCO annotation file: {jf.name}")
                    except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
                        continue
            elif fmt == "voc":
                xml_files = list(project.glob(f"**/{pattern}"))[:10]
                if xml_files:
                    indicators.append(f"{len(xml_files)}+ VOC XML annotation files")
            elif fmt == "yolo":
                # Only count .txt files that are alongside images
                txt_files = list(project.glob(f"**/{pattern}"))[:10]
                for tf in txt_files:
                    parent = tf.parent
                    has_images = any(
                        list(parent.glob(f"*{ext}"))[:1]
                        for ext in IMAGE_EXTENSIONS
                    )
                    if has_images:
                        indicators.append(f"YOLO annotation files in {parent.name}/")
                        break

    # Check requirements for CV packages
    for req_file in ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]:
        req_path = project / req_file
        if req_path.exists():
            try:
                content = req_path.read_text().lower()
                for pkg in CV_INDICATORS:
                    if pkg.lower() in content:
                        indicators.append(f"{pkg} in {req_file}")
            except (UnicodeDecodeError, PermissionError):
                continue

    # Check Python files for CV imports
    py_files = list(project.glob("**/*.py"))[:50]  # limit scan
    for py_file in py_files:
        try:
            content = py_file.read_text()
            for pkg in CV_INDICATORS:
                if f"import {pkg}" in content or f"from {pkg}" in content:
                    indicators.append(f"{pkg} import in {py_file.name}")
                    break
        except (UnicodeDecodeError, PermissionError):
            continue

    # Check for model files
    model_extensions = [".pt", ".pth", ".h5", ".onnx", ".tflite", ".savedmodel"]
    for ext in model_extensions:
        model_files = list(project.glob(f"**/*{ext}"))[:5]
        if model_files:
            indicators.append(f"{len(model_files)} {ext} model files found")

    return {
        "is_cv": len(indicators) > 0,
        "indicators": indicators,
    }


# --- Image Dataset Analysis ---

def analyze_image_dataset(dataset_path, annotation_format=None):
    """Analyze an image dataset: class distribution, resolution stats, quality checks.

    Args:
        dataset_path: root directory of the image dataset
        annotation_format: 'coco', 'voc', 'yolo', 'imagefolder', or None (auto-detect)

    Returns:
        dict with 'total_images', 'classes', 'class_distribution',
        'resolution_stats', 'annotation_format', 'issues'
    """
    dataset = Path(dataset_path)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Collect all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(dataset.glob(f"**/*{ext}"))
    image_files = sorted(image_files)

    if not image_files:
        return {
            "total_images": 0,
            "classes": [],
            "class_distribution": {},
            "resolution_stats": {},
            "annotation_format": None,
            "issues": ["No image files found in dataset path"],
        }

    # Auto-detect annotation format
    if annotation_format is None:
        annotation_format = _detect_annotation_format(dataset)

    # Class distribution
    class_distribution = _get_class_distribution(dataset, image_files, annotation_format)

    # Resolution statistics
    resolution_stats = _compute_resolution_stats(image_files)

    # Issues
    issues = []
    if class_distribution:
        counts = list(class_distribution.values())
        if max(counts) / max(min(counts), 1) > 5:
            issues.append(f"Class imbalance: ratio {max(counts)/max(min(counts),1):.1f}:1")

    classes = sorted(class_distribution.keys()) if class_distribution else []

    return {
        "total_images": len(image_files),
        "classes": classes,
        "class_distribution": class_distribution,
        "resolution_stats": resolution_stats,
        "annotation_format": annotation_format,
        "issues": issues,
    }


def _detect_annotation_format(dataset_path):
    """Auto-detect annotation format from dataset contents."""
    dataset = Path(dataset_path)

    # Check for COCO JSON
    json_files = list(dataset.glob("**/*.json"))
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
                if isinstance(data, dict) and "images" in data and "annotations" in data:
                    return "coco"
        except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
            continue

    # Check for VOC XML
    xml_files = list(dataset.glob("**/*.xml"))[:5]
    for xf in xml_files:
        try:
            content = xf.read_text()
            if "<annotation>" in content and "<object>" in content:
                return "voc"
        except (UnicodeDecodeError, PermissionError):
            continue

    # Check for YOLO TXT (lines with class_id x y w h)
    txt_files = list(dataset.glob("**/*.txt"))[:5]
    for tf in txt_files:
        if tf.name == "classes.txt":
            return "yolo"
        try:
            content = tf.read_text().strip()
            if content:
                parts = content.split("\n")[0].split()
                if len(parts) == 5 and all(_is_number(p) for p in parts):
                    return "yolo"
        except (UnicodeDecodeError, PermissionError):
            continue

    # Check for image-folder structure (class directories)
    subdirs = [d for d in dataset.iterdir() if d.is_dir()]
    has_images_in_subdirs = False
    for sd in subdirs[:10]:
        for ext in IMAGE_EXTENSIONS:
            if list(sd.glob(f"*{ext}"))[:1]:
                has_images_in_subdirs = True
                break
        if has_images_in_subdirs:
            break

    if has_images_in_subdirs:
        return "imagefolder"

    return None


def _is_number(s):
    """Check if string is a valid number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_class_distribution(dataset, image_files, annotation_format):
    """Get class distribution based on annotation format."""
    distribution = Counter()

    if annotation_format == "imagefolder":
        for img in image_files:
            class_name = img.parent.name
            distribution[class_name] += 1

    elif annotation_format == "coco":
        dataset = Path(dataset) if isinstance(dataset, str) else dataset
        json_files = list(dataset.glob("**/*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "annotations" in data and "categories" in data:
                        cat_map = {c["id"]: c["name"] for c in data["categories"]}
                        for ann in data["annotations"]:
                            cat_name = cat_map.get(ann["category_id"], f"unknown_{ann['category_id']}")
                            distribution[cat_name] += 1
            except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
                continue

    elif annotation_format == "yolo":
        classes_file = None
        for cf in dataset.glob("**/classes.txt") if isinstance(dataset, Path) else Path(dataset).glob("**/classes.txt"):
            classes_file = cf
            break

        class_names = {}
        if classes_file:
            try:
                lines = classes_file.read_text().strip().split("\n")
                class_names = {i: name.strip() for i, name in enumerate(lines)}
            except (UnicodeDecodeError, PermissionError):
                pass

        txt_files = list(Path(dataset).glob("**/*.txt")) if not isinstance(dataset, Path) else list(dataset.glob("**/*.txt"))
        for tf in txt_files:
            if tf.name == "classes.txt":
                continue
            try:
                for line in tf.read_text().strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cls_name = class_names.get(cls_id, f"class_{cls_id}")
                        distribution[cls_name] += 1
            except (ValueError, UnicodeDecodeError, PermissionError):
                continue

    elif annotation_format == "voc":
        xml_files = list(Path(dataset).glob("**/*.xml")) if not isinstance(dataset, Path) else list(dataset.glob("**/*.xml"))
        for xf in xml_files:
            try:
                content = xf.read_text()
                # Simple XML parsing for <name> tags inside <object>
                for match in re.finditer(r"<object>.*?<name>(.*?)</name>.*?</object>", content, re.DOTALL):
                    distribution[match.group(1)] += 1
            except (UnicodeDecodeError, PermissionError):
                continue

    return dict(distribution)


def _compute_resolution_stats(image_files):
    """Compute resolution statistics for image files.

    Uses PIL if available, otherwise returns empty stats.
    """
    widths = []
    heights = []
    channels = []
    corrupt_count = 0

    try:
        from PIL import Image
    except ImportError:
        return {
            "count": len(image_files),
            "note": "PIL not available -- install Pillow for resolution stats",
        }

    for img_path in image_files[:500]:  # limit for performance
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                mode = img.mode
                if mode == "RGB":
                    channels.append(3)
                elif mode == "RGBA":
                    channels.append(4)
                elif mode == "L":
                    channels.append(1)
                else:
                    channels.append(0)
        except Exception:
            corrupt_count += 1
            continue

    if not widths:
        return {"count": 0, "corrupt": corrupt_count}

    def _stats(values):
        n = len(values)
        sorted_v = sorted(values)
        return {
            "min": sorted_v[0],
            "max": sorted_v[-1],
            "mean": round(sum(values) / n, 1),
            "median": sorted_v[n // 2],
            "std": round((sum((v - sum(values) / n) ** 2 for v in values) / n) ** 0.5, 1),
        }

    return {
        "count": len(widths),
        "width": _stats(widths),
        "height": _stats(heights),
        "channels": dict(Counter(channels)),
        "corrupt": corrupt_count,
    }


# --- Augmentation Pipeline ---

def create_augmentation_pipeline(strategy="medium", task="classification"):
    """Create a data augmentation configuration for CV training.

    Args:
        strategy: 'light', 'medium', 'heavy', or dict of custom transforms
        task: 'classification', 'detection', or 'segmentation'

    Returns:
        dict with 'transforms' list and 'description'
    """
    if isinstance(strategy, dict):
        return {"transforms": strategy.get("transforms", []), "description": "Custom augmentation pipeline"}

    pipelines = {
        "light": {
            "transforms": [
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                {"name": "RandomRotation", "params": {"degrees": 10}},
                {"name": "ColorJitter", "params": {"brightness": 0.1, "contrast": 0.1}},
            ],
            "description": "Light augmentation: flip, small rotation, slight color jitter",
        },
        "medium": {
            "transforms": [
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                {"name": "RandomRotation", "params": {"degrees": 15}},
                {"name": "ColorJitter", "params": {
                    "brightness": 0.2, "contrast": 0.2,
                    "saturation": 0.2, "hue": 0.1
                }},
                {"name": "RandomResizedCrop", "params": {"scale": [0.8, 1.0]}},
                {"name": "RandomErasing", "params": {"p": 0.1}},
            ],
            "description": "Medium augmentation: flip, rotation, color jitter, crop, erasing",
        },
        "heavy": {
            "transforms": [
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                {"name": "RandomVerticalFlip", "params": {"p": 0.3}},
                {"name": "RandomRotation", "params": {"degrees": 30}},
                {"name": "ColorJitter", "params": {
                    "brightness": 0.4, "contrast": 0.4,
                    "saturation": 0.4, "hue": 0.2
                }},
                {"name": "RandomAffine", "params": {
                    "degrees": 15, "translate": [0.1, 0.1],
                    "scale": [0.8, 1.2], "shear": 10
                }},
                {"name": "GaussianBlur", "params": {"kernel_size": 3, "p": 0.2}},
                {"name": "RandomErasing", "params": {"p": 0.2}},
            ],
            "description": "Heavy augmentation: aggressive spatial and color transforms",
        },
    }

    if strategy not in pipelines:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'light', 'medium', 'heavy', or custom dict.")

    pipeline = pipelines[strategy]

    # Add bbox-safe warning for detection tasks
    if task == "detection":
        pipeline["note"] = "Detection task: ensure transforms update bounding boxes (use albumentations or torchvision v2)"

    return pipeline


# --- CV Metrics ---

def compute_cv_metrics(predictions, ground_truth, task="classification",
                       num_classes=None, class_names=None):
    """Compute computer vision evaluation metrics.

    Args:
        predictions: model predictions
            - classification: list of predicted class indices
            - detection: list of dicts with 'boxes', 'labels', 'scores'
            - segmentation: list of 2D arrays (predicted masks)
        ground_truth: ground truth labels
            - classification: list of true class indices
            - detection: list of dicts with 'boxes', 'labels'
            - segmentation: list of 2D arrays (ground truth masks)
        task: 'classification', 'detection', or 'segmentation'
        num_classes: number of classes (auto-detected if None)
        class_names: list of class name strings (optional)

    Returns:
        dict with task-appropriate metrics
    """
    if task == "classification":
        return _compute_classification_metrics(predictions, ground_truth, num_classes, class_names)
    elif task == "detection":
        return _compute_detection_metrics(predictions, ground_truth, num_classes, class_names)
    elif task == "segmentation":
        return _compute_segmentation_metrics(predictions, ground_truth, num_classes, class_names)
    else:
        raise ValueError(f"Unknown task: {task}")


def _compute_classification_metrics(predictions, ground_truth, num_classes=None, class_names=None):
    """Compute classification metrics: accuracy, precision, recall, F1, confusion matrix."""
    assert len(predictions) == len(ground_truth), "Prediction/ground truth count mismatch"

    if num_classes is None:
        num_classes = max(max(predictions), max(ground_truth)) + 1

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Overall accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions)

    # Confusion matrix
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for p, g in zip(predictions, ground_truth):
        if 0 <= p < num_classes and 0 <= g < num_classes:
            confusion[g][p] += 1

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(num_classes)) - tp
        fn = sum(confusion[c]) - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[class_names[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[c]),
        }

    # Macro averages
    macro_precision = sum(m["precision"] for m in per_class.values()) / num_classes
    macro_recall = sum(m["recall"] for m in per_class.values()) / num_classes
    macro_f1 = sum(m["f1"] for m in per_class.values()) / num_classes

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_names": class_names,
    }


def _compute_detection_metrics(predictions, ground_truth, num_classes=None, class_names=None):
    """Compute detection metrics: mAP at various IoU thresholds.

    Simplified mAP computation using IoU matching.
    """
    assert len(predictions) == len(ground_truth), "Prediction/ground truth count mismatch"

    iou_thresholds = [0.5, 0.75]

    # Collect all detections and ground truths by class
    if num_classes is None:
        all_labels = set()
        for gt in ground_truth:
            all_labels.update(gt.get("labels", []))
        for pred in predictions:
            all_labels.update(pred.get("labels", []))
        num_classes = max(all_labels) + 1 if all_labels else 0

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    results = {}
    for iou_thresh in iou_thresholds:
        per_class_ap = {}
        for cls_id in range(num_classes):
            ap = _compute_ap_for_class(predictions, ground_truth, cls_id, iou_thresh)
            per_class_ap[class_names[cls_id]] = round(ap, 4)

        ap_values = list(per_class_ap.values())
        mean_ap = sum(ap_values) / max(len(ap_values), 1)
        results[f"mAP@{iou_thresh}"] = round(mean_ap, 4)
        results[f"per_class_AP@{iou_thresh}"] = per_class_ap

    return results


def _compute_ap_for_class(predictions, ground_truth, cls_id, iou_threshold):
    """Compute Average Precision for a single class at given IoU threshold."""
    # Collect all detections for this class across all images
    all_detections = []  # (confidence, is_tp)
    total_gt = 0

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        pred_boxes = pred.get("boxes", [])
        pred_labels = pred.get("labels", [])
        pred_scores = pred.get("scores", [])

        gt_boxes = gt.get("boxes", [])
        gt_labels = gt.get("labels", [])

        # Filter for target class
        cls_pred_indices = [i for i, l in enumerate(pred_labels) if l == cls_id]
        cls_gt_indices = [i for i, l in enumerate(gt_labels) if l == cls_id]

        total_gt += len(cls_gt_indices)
        matched_gt = set()

        # Sort predictions by confidence
        cls_preds = [(pred_scores[i], pred_boxes[i]) for i in cls_pred_indices]
        cls_preds.sort(key=lambda x: x[0], reverse=True)

        for score, pred_box in cls_preds:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx in cls_gt_indices:
                if gt_idx in matched_gt:
                    continue
                iou = _compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_detections.append((score, True))
                matched_gt.add(best_gt_idx)
            else:
                all_detections.append((score, False))

    if total_gt == 0:
        return 0.0

    # Sort by confidence and compute precision-recall curve
    all_detections.sort(key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for score, is_tp in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using all-point interpolation
    ap = 0.0
    for i in range(len(precisions)):
        if i == 0:
            ap += precisions[i] * recalls[i]
        else:
            ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap


def _compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-8)


def _compute_segmentation_metrics(predictions, ground_truth, num_classes=None, class_names=None):
    """Compute segmentation metrics: mIoU, per-class IoU, pixel accuracy."""
    assert len(predictions) == len(ground_truth), "Prediction/ground truth count mismatch"

    if num_classes is None:
        all_values = set()
        for gt in ground_truth:
            if hasattr(gt, "flatten"):
                all_values.update(gt.flatten().tolist())
            else:
                for row in gt:
                    all_values.update(row)
        num_classes = max(all_values) + 1 if all_values else 0

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Accumulate intersection and union per class
    intersection_per_class = [0] * num_classes
    union_per_class = [0] * num_classes
    total_correct = 0
    total_pixels = 0

    for pred, gt in zip(predictions, ground_truth):
        # Convert to flat lists if needed
        if hasattr(pred, "flatten"):
            pred_flat = pred.flatten().tolist()
            gt_flat = gt.flatten().tolist()
        else:
            pred_flat = [p for row in pred for p in row]
            gt_flat = [g for row in gt for g in row]

        for p, g in zip(pred_flat, gt_flat):
            if p == g:
                total_correct += 1
            total_pixels += 1

            if 0 <= g < num_classes:
                if p == g:
                    intersection_per_class[g] += 1
                union_per_class[g] += 1
            if 0 <= p < num_classes and p != g:
                union_per_class[p] += 1

    # Per-class IoU
    per_class_iou = {}
    valid_ious = []
    for c in range(num_classes):
        if union_per_class[c] > 0:
            iou = intersection_per_class[c] / union_per_class[c]
            per_class_iou[class_names[c]] = round(iou, 4)
            valid_ious.append(iou)
        else:
            per_class_iou[class_names[c]] = None

    mean_iou = sum(valid_ious) / max(len(valid_ious), 1)
    pixel_accuracy = total_correct / max(total_pixels, 1)

    return {
        "mIoU": round(mean_iou, 4),
        "pixel_accuracy": round(pixel_accuracy, 4),
        "per_class_iou": per_class_iou,
        "class_names": class_names,
    }


# --- ONNX Export ---

def export_to_onnx(model, dummy_input, output_path, input_names=None,
                   output_names=None, dynamic_axes=None, opset_version=17):
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model (nn.Module)
        dummy_input: example input tensor for tracing
        output_path: path to save .onnx file
        input_names: list of input tensor names (default: ['input'])
        output_names: list of output tensor names (default: ['output'])
        dynamic_axes: dict of dynamic axis specifications
        opset_version: ONNX opset version (default: 17)

    Returns:
        dict with 'output_path', 'model_size_mb', 'opset_version', 'valid'
    """
    try:
        import torch
        import torch.onnx
    except ImportError:
        raise ImportError("PyTorch required for ONNX export. Install with: pip install torch")

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate
    valid = False
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        valid = True
    except ImportError:
        pass  # onnx package not installed, skip validation
    except Exception:
        valid = False

    model_size_mb = output_path.stat().st_size / (1024 * 1024)

    return {
        "output_path": str(output_path),
        "model_size_mb": round(model_size_mb, 2),
        "opset_version": opset_version,
        "valid": valid,
    }


# --- DataLoader Creation ---

def create_image_dataloader(dataset_path, target_size=(224, 224), batch_size=32,
                            shuffle=True, augmentation=None, normalize="imagenet",
                            num_workers=4):
    """Create a PyTorch DataLoader for an image-folder dataset.

    Args:
        dataset_path: path to dataset (class-per-folder structure)
        target_size: (height, width) to resize images
        batch_size: batch size
        shuffle: whether to shuffle
        augmentation: augmentation pipeline dict from create_augmentation_pipeline()
        normalize: 'imagenet', 'minmax', or dict with 'mean' and 'std'
        num_workers: number of data loading workers

    Returns:
        dict with 'dataloader', 'dataset', 'num_classes', 'class_names', 'dataset_size'
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError(
            "PyTorch and torchvision required. "
            "Install with: pip install torch torchvision"
        )

    # Build transform list
    transform_list = []

    # Resize
    transform_list.append(transforms.Resize(target_size))

    # Augmentation transforms
    if augmentation and "transforms" in augmentation:
        for t in augmentation["transforms"]:
            name = t["name"]
            params = t.get("params", {})
            if hasattr(transforms, name):
                transform_cls = getattr(transforms, name)
                try:
                    transform_list.append(transform_cls(**params))
                except TypeError:
                    pass  # skip transforms with incompatible params

    # ToTensor
    transform_list.append(transforms.ToTensor())

    # Normalization
    if normalize == "imagenet":
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif normalize == "minmax":
        pass  # ToTensor already scales to [0, 1]
    elif isinstance(normalize, dict):
        transform_list.append(
            transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
        )

    composed = transforms.Compose(transform_list)

    # Create dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=composed)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "dataloader": dataloader,
        "dataset": dataset,
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "dataset_size": len(dataset),
    }
