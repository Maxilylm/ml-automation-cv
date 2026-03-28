# /cv-preprocess

Image preprocessing pipeline: resize, normalize, augment, and split into train/val/test sets.

## Usage

```
/cv-preprocess <dataset_path> [--target-size 224x224] [--split 0.7,0.15,0.15] [--normalize imagenet] [--output processed/]
```

- `dataset_path`: root directory of the image dataset
- `--target-size`: target image dimensions (default: 224x224)
- `--split`: train/val/test split ratios (default: 0.7,0.15,0.15)
- `--normalize`: normalization scheme (imagenet, minmax, standardize, default: imagenet)
- `--output`: output directory (default: `processed/`)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify dataset path exists and contains images
4. Validate target size format and split ratios sum to 1.0

### Stage 1: Dataset Analysis

1. Scan dataset for all image files
2. Detect annotation format (COCO, VOC, YOLO, or image-folder)
3. Compute current resolution statistics
4. Report: total images, current resolution range, annotation format

### Stage 2: Resize

1. Resize all images to `--target-size` preserving aspect ratio (pad or letterbox)
2. Interpolation: LANCZOS for downscaling, BICUBIC for upscaling
3. Update annotations (scale bounding boxes, resize segmentation masks)
4. Report: images resized, padding applied

### Stage 3: Normalization

1. Compute dataset mean and std per channel (if `--normalize standardize`)
2. Apply normalization:
   - `imagenet`: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - `minmax`: scale pixel values to [0, 1]
   - `standardize`: per-dataset mean/std normalization
3. Save normalization parameters to `config/normalize_params.json`

### Stage 4: Train/Val/Test Split

1. Split dataset using stratified sampling (preserve class ratios)
2. Create directory structure: `{output}/train/`, `{output}/val/`, `{output}/test/`
3. Copy/move images and annotations to split directories
4. Verify split integrity (no data leakage, correct ratios)
5. Report: split counts and class distribution per split

### Stage 5: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-analyst", {
    "status": "completed",
    "target_size": target_size,
    "normalization": normalize_scheme,
    "split_ratios": split_ratios,
    "split_counts": {"train": train_count, "val": val_count, "test": test_count},
    "normalization_params": norm_params,
    "output_path": output_path,
    "recommendations": recommendations
})
```

Write report to `reports/cv_preprocessing_report.json`.

Print summary: target size, normalization, split counts per class.
