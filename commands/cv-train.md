# /cv-train

Train a computer vision model for classification, detection, or segmentation with transfer learning.

## Usage

```
/cv-train <dataset_path> [--task classification|detection|segmentation] [--backbone resnet50] [--epochs 50] [--lr 0.001] [--batch-size 32] [--framework pytorch|tensorflow]
```

- `dataset_path`: preprocessed dataset directory (with train/val splits)
- `--task`: CV task type (default: auto-detect from annotations)
- `--backbone`: pretrained backbone (default: resnet50)
- `--epochs`: training epochs (default: 50)
- `--lr`: initial learning rate (default: 0.001)
- `--batch-size`: batch size (default: 32)
- `--framework`: deep learning framework (default: pytorch)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` -- if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `cv_utils.py` exists in `src/` -- if missing, copy from this plugin's `templates/cv_utils.py`
3. Verify dataset has train/val splits
4. Check GPU availability (`torch.cuda.is_available()` or `tf.config.list_physical_devices('GPU')`)
5. Auto-detect task type from annotation format if not specified

### Stage 1: Data Loading

1. Create dataset class with appropriate transforms:
   - **Classification**: `ImageFolder` or custom dataset with labels
   - **Detection**: dataset returning images + bounding boxes + class labels
   - **Segmentation**: dataset returning images + segmentation masks
2. Configure data augmentation (from `cv_utils.create_augmentation_pipeline()`)
3. Create DataLoaders with prefetching, num_workers, pin_memory
4. Report: dataset size per split, augmentation pipeline, batch count

### Stage 2: Model Architecture

1. Load pretrained backbone from model zoo
2. Modify head for target task:
   - **Classification**: replace FC layer with `nn.Linear(features, num_classes)`
   - **Detection**: configure anchor boxes, FPN neck, detection head
   - **Segmentation**: attach decoder (U-Net decoder, ASPP module)
3. Freeze backbone (optional, based on dataset size)
4. Report: model architecture, total parameters, trainable parameters

### Stage 3: Training Configuration

1. **Optimizer**: Adam/AdamW with weight decay
2. **Scheduler**: cosine annealing with warmup (5% of epochs)
3. **Loss function**:
   - Classification: CrossEntropyLoss (or FocalLoss for imbalanced)
   - Detection: combination of classification + box regression + objectness
   - Segmentation: DiceLoss + CrossEntropyLoss
4. **Mixed precision**: enable AMP if GPU available
5. **Callbacks**: early stopping (patience=10), checkpoint saving
6. Generate `config/training_config.json`

### Stage 4: Training Loop

1. Train for specified epochs with validation after each epoch
2. Log per-epoch: train loss, val loss, primary metric (accuracy/mAP/mIoU)
3. Save best model checkpoint based on validation metric
4. Save periodic checkpoints every 10 epochs
5. Generate training curves (loss and metric vs. epoch)

### Stage 5: Final Evaluation

1. Load best checkpoint
2. Evaluate on validation set:
   - **Classification**: accuracy, precision, recall, F1 per class
   - **Detection**: mAP@0.5, mAP@0.5:0.95 per class
   - **Segmentation**: mIoU, per-class IoU, pixel accuracy
3. Generate confusion matrix (classification) or PR curves (detection)

### Stage 6: Report

```python
from ml_utils import save_agent_report
save_agent_report("cv-engineer", {
    "status": "completed",
    "task": task,
    "backbone": backbone,
    "framework": framework,
    "epochs_trained": epochs_trained,
    "best_epoch": best_epoch,
    "best_metric": best_metric_value,
    "training_config": training_config,
    "metrics": final_metrics,
    "checkpoint_path": best_checkpoint_path,
    "recommendations": recommendations
})
```

Write report to `reports/cv_training_report.json`.

Print: task, backbone, best metric, checkpoint path, training time.
