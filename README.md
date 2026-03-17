# Patch Range Transformer (PRT) — Brain Tumor MRI Classification

> A novel vision transformer architecture built entirely from scratch — no pretrained weights.

---

## Result

| Metric | Value |
|--------|-------|
| Test Accuracy | **91.44%** |
| Parameters | 14.6M |
| Training images | 5,600 |
| Test images | 1,600 |
| Classes | 4 |
| Epochs | 100 |

---

## Architecture — What makes PRT different

| Model | Attention Style |
|-------|----------------|
| ViT | Every patch attends to every patch (full global) |
| Swin | Patches attend within fixed window only (local) |
| **PRT (ours)** | **Patches attend within radius R + CLS bridges globally** |
```
Each patch token  →  attends to neighbors within Chebyshev radius R=3 (49 patches max)
CLS token         →  attends to ALL patches (global aggregator)
All patches       →  can attend back to CLS token
Final prediction  →  CLS output + mean of all patch outputs (fused)
```

---

## Training Curves

![Training Curves](prt_training_curves.png)

---

## Per-Class Results

![Classification Report](prt_classification_report.png)


---

## Confusion Matrix

![Confusion Matrix](prt_confusion_matrix.png)

---

## Model Config
```python
IMG_SIZE   = 224
PATCH_SIZE = 16       # 14x14 = 196 patches
EMBED_DIM  = 384
NUM_HEADS  = 12
NUM_LAYERS = 8
RANGE_R    = 3        # each patch attends to 49 spatial neighbors
EPOCHS     = 100
LR         = 3e-4     # cosine decay with 5-epoch linear warmup
LOSS       = FocalLoss(gamma=2.0, label_smoothing=0.1)
```

---

## Key Findings

- No Tumor and Pituitary classified near-perfectly — 100% and 98% recall
- Glioma is the hardest class (75.5% recall) — visually similar to meningioma
- CLS + mean pooling fusion outperforms CLS-only by ~2%
- LR warmup (5 epochs linear) was critical — without it model diverged early
- Bigger model (32M params) performed worse — 14.6M is the right size for 5,600 images

---

## Dataset

**Brain Tumor MRI** by Masoud Nickparvar  
Source: [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
4 classes: Glioma · Meningioma · No Tumor · Pituitary

---

## Stack

PyTorch · einops · scikit-learn · Tesla P100 · Kaggle Notebooks

---

## Notebook

[View full notebook on Kaggle](YOUR_KAGGLE_URL_HERE)
