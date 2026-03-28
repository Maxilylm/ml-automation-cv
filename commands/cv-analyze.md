# /cv-analyze

Image dataset EDA with class balance, resolution distribution, quality metrics, and sample grid.

## Usage

```
/cv-analyze <dataset_path> [--format coco|voc|yolo|imagefolder] [--output reports/]
```

- `dataset_path`: root directory of the image dataset
- `--format`: annotation format (default: auto-detect)
- `--output`: output directory for reports (default: `reports/`)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify dataset path exists and contains images
4. Auto-detect annotation format if not specified

### Stage 1: Dataset Inventory

1. Scan dataset directory for image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`)
2. Detect directory structure (flat, class-per-folder, train/val/test splits)
3. Load annotation files if present (COCO JSON, VOC XML, YOLO TXT)
4. Report: total images, total classes, annotation format, directory structure

### Stage 2: Class Distribution

1. Count images per class (from folder names or annotations)
2. Compute imbalance ratio (majority / minority class count)
3. Identify under-represented classes (< 5% of largest class)
4. Recommend resampling strategy if imbalance ratio > 5:1
5. Generate class distribution bar chart

### Stage 3: Resolution Statistics

1. Read dimensions of all images (width, height, channels)
2. Compute: min, max, mean, median, std for width and height
3. Identify outliers (> 2 std from mean)
4. Compute aspect ratio distribution
5. Recommend target resolution for training

### Stage 4: Quality Assessment

1. **Blur detection** -- Laplacian variance per image, flag below threshold
2. **Duplicate detection** -- perceptual hash (pHash), flag pairs with hamming distance < 5
3. **Corrupt file detection** -- attempt to open each image, flag failures
4. **Color space check** -- verify consistent channels (RGB vs grayscale)
5. Report: blurry count, duplicate pairs, corrupt files, mixed color spaces

### Stage 5: Label Verification (if annotations exist)

1. Cross-reference annotations with image files:
   - Images without annotations (unlabeled)
   - Annotations without corresponding images (orphaned)
2. Validate bounding boxes (within image bounds, non-zero area)
3. Validate segmentation masks (valid polygons, non-empty)
4. Flag suspicious annotations (extremely small/large boxes)

### Stage 6: Sample Grid

1. Select representative images per class (random or stratified)
2. Generate sample grid visualization
3. If annotations: overlay bounding boxes / masks on sample images

### Stage 7: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-analyst", {
    "status": "completed",
    "dataset_path": dataset_path,
    "total_images": total_images,
    "total_classes": num_classes,
    "annotation_format": format,
    "class_distribution": class_counts,
    "imbalance_ratio": imbalance_ratio,
    "resolution_stats": {"min": min_res, "max": max_res, "mean": mean_res, "median": median_res},
    "quality_issues": {"blurry": blurry_count, "duplicates": dup_count, "corrupt": corrupt_count},
    "label_issues": label_issues,
    "recommendations": recommendations
})
```

Write report to `reports/cv_dataset_analysis.json`.

Print summary table with dataset statistics, quality issues, and recommendations.
