# 🧠 Alzheimer’s Disease Staging Using a Dual-Stream CNN–Transformer Architecture

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Transformers](https://img.shields.io/badge/Architecture-CNN%20%2B%20ViT-green.svg)
![Status](https://img.shields.io/badge/Status-Active%20Research-yellow.svg)

---

## 1️⃣ Abstract

Accurate staging of Alzheimer’s Disease (AD) using structural MRI remains challenging due to:

- Subtle anatomical variations  
- Inter-subject heterogeneity  
- Severe class imbalance  

Many prior works report high accuracy but suffer from:

- Slice-level evaluation  
- Data leakage  
- Limited generalization analysis  

This project develops a **subject-level dual-stream deep learning framework** integrating Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for robust multi-class Alzheimer’s staging.

The model is evaluated under strict **patient-wise experimental conditions** to ensure methodological rigor and reproducibility.

### 🎯 Target Classes

- **CN** – Cognitively Normal  
- **LMCI** – Late Mild Cognitive Impairment  
- **AD** – Alzheimer’s Disease  

---

## 2️⃣ Dataset

- **Source:** ADNI-derived structural MRI dataset  
- **Total Subjects:** 639  

| Class | Subjects |
|--------|----------|
| CN     | 195      |
| LMCI   | 311      |
| AD     | 133      |

### 🔒 Key Protocol Decisions

- Strict patient-wise splitting  
- Fixed number of slices per subject  
- No augmented subject duplication  
- No slice-level leakage  

> ⚠️ The dataset is not included in this repository due to licensing and storage constraints.

---

## 3️⃣ Research Motivation

Existing literature commonly demonstrates:

- High performance using slice-level splits  
- Heavy emphasis on binary classification (AD vs CN)  
- Limited exploration of robust deep feature fusion  
- Insufficient methodological validation  

### This Work Addresses These Limitations Through:

- Subject-level modeling  
- Dual-stream representation learning  
- Controlled architectural comparisons  
- Structured feature fusion experiments  

---

## 4️⃣ Methodology

### 🔹 Phase 1 — Exploratory Analysis & Baselines

- CNN feature extraction (ResNet50)  
- Classical classifiers (SVM / Logistic Regression)  
- 5-fold cross-validation  

---

### 🔹 Phase 2 — Transformer Integration

- ViT-B/16 backbone (ImageNet pretrained)  
- Subject-level feature aggregation  
- Comparative evaluation with CNN baseline  

---

### 🔹 Phase 3 — Feature Fusion Experiments

- CNN-only evaluation  
- ViT-only evaluation  
- CNN + ViT concatenation baseline  

---

### 🔹 Phase 4 — Efficient Tensor Dataset Construction

- Migration from PNG slice loading to tensor-based dataset  
- GPU-optimized batching  
- Stabilized training pipeline  

---

### 🔹 Phase 5 — Dual-Stream End-to-End Architecture

- Parallel CNN and ViT feature extractors  
- Feature-level fusion layer  
- Class-weighted cross-entropy optimization  
- Partial fine-tuning strategy  

---

## 5️⃣ Preliminary Results

### 📊 Frozen Feature Baselines (5-Fold Cross-Validation)

| Model | Accuracy | Macro-F1 |
|--------|----------|----------|
| CNN (ResNet50) | ~0.58 | ~0.56 |
| ViT-B/16 | ~0.42 | ~0.40 |
| CNN + ViT (Concat) | ~0.57 | ~0.55 |

---

### ⚡ End-to-End Dual-Stream Model (Partial Fine-Tuning)

- Stable training achieved  
- Validation Macro-F1 ≈ 0.46  
- Identified slice-level aggregation as a key performance bottleneck  

These findings motivate the exploration of:

- Attention-based aggregation  
- Improved fusion mechanisms  
- Representation alignment strategies  

---

## 6️⃣ Experimental Rigor

- Strict patient-wise data splitting  
- No data leakage  
- Class imbalance handled via weighted loss  
- Controlled baseline comparisons  
- Transparent reporting of intermediate performance  

This project emphasizes **methodological integrity over inflated metrics**.

---

## 7️⃣ Ongoing Work

- Attention-based slice aggregation  
- Contrastive embedding alignment  
- Advanced dual-stream fusion mechanisms  
- External validation on independent datasets  
- Explainability via Grad-CAM and transformer attention rollout  

---

## 8️⃣ Reproducibility

### Environment

- Python 3.9+  
- PyTorch  
- timm  
- scikit-learn  
- NumPy  
- Pandas  

All experiments are reproducible given access to the dataset and preprocessing pipeline.

---

## 9️⃣ Research Status

This project is under active development as part of an academic research study focused on robust MRI-based Alzheimer’s staging using **dual-stream representation learning**.

The current direction prioritizes:

- Generalization robustness  
- Methodological transparency  
- Clinically meaningful multi-class discrimination  
