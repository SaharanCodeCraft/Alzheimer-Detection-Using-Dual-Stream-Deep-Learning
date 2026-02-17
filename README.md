# Alzheimer’s Disease Staging Using a Dual-Stream CNN–Transformer Architecture

## 1. Abstract

Accurate staging of Alzheimer’s Disease (AD) using structural MRI remains challenging due to subtle anatomical variations, inter-subject heterogeneity, and class imbalance. Many prior works report high accuracy but suffer from slice-level evaluation, data leakage, or limited generalization analysis.

This project develops a subject-level dual-stream deep learning framework that integrates convolutional neural networks (CNNs) and Vision Transformers (ViTs) for robust multi-class Alzheimer’s staging. The model is evaluated under strict patient-wise conditions to ensure methodological rigor and reproducibility.

Target Classes:
- CN – Cognitively Normal  
- LMCI – Late Mild Cognitive Impairment  
- AD – Alzheimer’s Disease  

---

## 2. Dataset

- Source: ADNI-derived structural MRI dataset  
- Total Subjects: 639  
  - CN: 195  
  - LMCI: 311  
  - AD: 133  

Key Protocol Decisions:
- Strict patient-wise splitting  
- Fixed number of slices per subject  
- No augmented subject duplication  
- No slice-level leakage  

The dataset is not included in this repository due to licensing and storage constraints.

---

## 3. Research Motivation

Existing literature demonstrates:

- High performance often reported using slice-level splits  
- Heavy emphasis on binary classification (AD vs CN)  
- Limited exploration of robust deep feature fusion  
- Insufficient methodological validation  

This work addresses these limitations through:
- Subject-level modeling  
- Dual-stream representation learning  
- Controlled architectural comparisons  
- Structured feature fusion experiments  

---

## 4. Methodology

### Phase 1 — Exploratory Analysis & Baselines
- CNN feature extraction (ResNet50)
- Classical classifiers (SVM / Logistic Regression)
- 5-fold cross-validation

### Phase 2 — Transformer Integration
- ViT-B/16 backbone (ImageNet pretrained)
- Subject-level feature aggregation
- Comparative evaluation with CNN baseline

### Phase 3 — Feature Fusion Experiments
- CNN-only evaluation
- ViT-only evaluation
- CNN + ViT concatenation baseline

### Phase 4 — Efficient Tensor Dataset Construction
- Conversion from PNG slice loading to tensor-based dataset
- GPU-optimized batching
- Stabilized training pipeline

### Phase 5 — Dual-Stream End-to-End Architecture
- Parallel CNN and ViT feature extractors
- Feature-level fusion layer
- Class-weighted cross-entropy optimization
- Partial fine-tuning strategy

---

## 5. Preliminary Results

Frozen Feature Baselines (5-fold Cross-Validation):

| Model | Accuracy | Macro-F1 |
|--------|----------|----------|
| CNN (ResNet50) | ~0.58 | ~0.56 |
| ViT-B/16 | ~0.42 | ~0.40 |
| CNN + ViT (Concat) | ~0.57 | ~0.55 |

End-to-End Dual-Stream Model (Partial Fine-Tuning):

- Stable training achieved
- Validation Macro-F1 ≈ 0.46
- Identified slice-level aggregation as a key performance bottleneck

These findings motivate the exploration of attention-based aggregation and improved fusion mechanisms.

---

## 6. Experimental Rigor

- Strict patient-wise data splitting  
- No data leakage  
- Class imbalance handled via weighted loss  
- Controlled baseline comparisons  
- Transparent reporting of intermediate performance  

---

## 7. Ongoing Work

- Attention-based slice aggregation  
- Contrastive embedding alignment  
- Improved dual-stream fusion mechanisms  
- External validation on independent datasets  
- Explainability via Grad-CAM and transformer attention rollout  

---

## 8. Reproducibility

Environment:
- Python 3.9+
- PyTorch
- timm
- scikit-learn
- NumPy
- Pandas

All experiments are reproducible given access to the dataset and preprocessing pipeline.

---

## 9. Research Status

This project is under active development as part of an academic research study focused on robust MRI-based Alzheimer’s staging using dual-stream representation learning.
