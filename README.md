# 🌿 Plant Disease Classification Using Traditional Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, scientifically rigorous Computer Vision and Traditional Machine Learning system for automated diagnosis and classification of plant leaf diseases across Pepper, Potato, and Tomato crops.

**Strict Design Constraint:** Built entirely using **Traditional Computer Vision descriptors and classical Machine Learning algorithms** (Color Histograms, Local Binary Patterns, Gray-Level Co-occurrence Matrix, Histogram of Oriented Gradients, Support Vector Machines, and Random Forest). **Zero Deep Learning frameworks (TensorFlow, PyTorch, Keras, CNNs, or ResNets) are used.**

---

## 📋 Table of Contents
1. [Project Introduction & Background](#1-project-introduction--background)
2. [Project Purpose & Objectives](#2-project-purpose--objectives)
3. [Dataset & Actual Verified Statistics](#3-dataset--actual-verified-statistics)
4. [Features & Target Variables](#4-features--target-variables)
5. [Image Preprocessing Methodology](#5-image-preprocessing-methodology)
6. [Feature Engineering & Representation Ablation](#6-feature-engineering--representation-ablation)
7. [Traditional Machine Learning Algorithms](#7-traditional-machine-learning-algorithms)
8. [Machine Learning Pipelines & Data Leakage Prevention](#8-machine-learning-pipelines--data-leakage-prevention)
9. [Dataset Splitting & Stratification](#9-dataset-splitting--stratification)
10. [Feature Scaling](#10-feature-scaling)
11. [Hyperparameter Optimization](#11-hyperparameter-optimization)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Experimental Results & Model Comparison](#13-experimental-results--model-comparison)
14. [Project Architecture](#14-project-architecture)
15. [Installation & Setup](#15-installation--setup)
16. [Training Instructions](#16-training-instructions)
17. [Streamlit Web Application](#17-streamlit-web-application)
18. [Prediction Instructions (CLI & Python API)](#18-prediction-instructions)
19. [Jupyter Notebooks Guide](#19-jupyter-notebooks-guide)
20. [Google Colab Workflow](#20-google-colab-workflow)
21. [Automated Testing & Quality Assurance](#21-automated-testing--quality-assurance)
22. [Model Limitations & Threat to Validity](#22-model-limitations--threat-to-validity)
23. [Future Improvements](#23-future-improvements)

---

## 1. Project Introduction & Background

Foliar plant diseases are among the primary threats to agricultural yields, farm revenues, and global food security. Bacterial infections, fungal blights, viral deformities, and mite damage cause rapid chlorosis (tissue yellowing) and necrosis (tissue death). If left undetected, pathogens spread rapidly across entire fields.

Visual inspection by agricultural agronomists is labor-intensive, subjective, and prone to error—especially when disparate pathogens present overlapping visual symptoms (e.g. *Alternaria solani* early blight vs *Phytophthora infestans* late blight).

While contemporary literature often turns directly to heavy deep convolutional neural networks requiring costly GPU clusters and opaque black-box parameters, this project proves that **multi-modal handcrafted Computer Vision feature engineering paired with Scikit-learn Machine Learning pipelines** achieves exceptional accuracy (**95.37%** with SVM), trains in **seconds on commodity CPU hardware**, and maintains fully transparent, interpretable visual representations.

---

## 2. Project Purpose & Objectives

### Purpose
To develop, validate, compare, and deploy a traditional Machine Learning image classification system capable of diagnosing plant leaf diseases from RGB photographs across Solanaceae family crops (Pepper, Potato, and Tomato).

### Core Objectives
1. **Dataset Ingestion & Auditing:** Verify, preserve, and explore the PlantVillage dataset across 15 healthy and diseased leaf categories.
2. **Standardized Preprocessing:** Build deterministic image loading, format normalization, spatial resizing ($128 \times 128$), and array bounds verification.
3. **Multi-Modal Feature Engineering:** Extract color distributions (RGB and HSV histograms), surface micro-textures (LBP and GLCM Haralick statistics), gradient structures (HOG), and conduct diagnostic shape analysis.
4. **Leak-Free ML Pipelines:** Implement Scikit-learn Pipelines encapsulating `StandardScaler`, `SVC` (RBF kernel), and `RandomForestClassifier`.
5. **Cross-Validated Hyperparameter Tuning:** Optimize hyperparameters on training data using Stratified K-Fold cross-validation focused on Macro F1-score.
6. **Empirical Evaluation & Model Selection:** Benchmark both models using Accuracy, Macro/Weighted Precision, Recall, F1, Confusion Matrices, and identify the superior model.
7. **End-to-End Application & Deployment:** Deliver an interactive Streamlit web dashboard, standalone CLI inference, Jupyter notebooks, and a Google Colab notebook.

---

## 3. Dataset & Actual Verified Statistics

- **Dataset Name:** PlantVillage Dataset (Crop Disease Classification)
- **Source:** Kaggle ([Emma Rex Plant Disease Collection](https://www.kaggle.com/datasets/emmarex/plantdisease))
- **Local Location:** `data/raw/PlantVillage/`
- **Total Valid Images:** **20,638**
- **Total Classes:** **15** (12 pathology categories + 3 healthy baselines)
- **Total Host Species:** **3** (Pepper, Potato, Tomato)
- **Image Format:** RGB JPEG images (native resolution $256 \times 256$ pixels, verified 100% uncorrupted)

### Complete Verified Class Frequencies & Imbalance Distribution

| Index | Class Label | Host Species | Condition / Pathology | Image Count | Percentage (%) |
| :---: | :--- | :---: | :--- | :---: | :---: |
| 1 | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato | Tomato Yellow Leaf Curl Virus | 3,208 | 15.54% |
| 2 | `Tomato_Bacterial_spot` | Tomato | Bacterial Spot (*Xanthomonas*) | 2,127 | 10.31% |
| 3 | `Tomato_Late_blight` | Tomato | Late Blight (*Phytophthora infestans*) | 1,909 | 9.25% |
| 4 | `Tomato_Septoria_leaf_spot` | Tomato | Septoria Leaf Spot (*Septoria lycopersici*) | 1,771 | 8.58% |
| 5 | `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomato | Two-Spotted Spider Mite (*Tetranychus urticae*) | 1,676 | 8.12% |
| 6 | `Tomato_healthy` | Tomato | Healthy Foliage (Baseline) | 1,591 | 7.71% |
| 7 | `Pepper__bell___healthy` | Pepper | Healthy Foliage (Baseline) | 1,478 | 7.16% |
| 8 | `Tomato__Target_Spot` | Tomato | Target Spot (*Corynespora cassiicola*) | 1,404 | 6.80% |
| 9 | `Potato___Early_blight` | Potato | Early Blight (*Alternaria solani*) | 1,000 | 4.85% |
| 10 | `Potato___Late_blight` | Potato | Late Blight (*Phytophthora infestans*) | 1,000 | 4.85% |
| 11 | `Tomato_Early_blight` | Tomato | Early Blight (*Alternaria solani*) | 1,000 | 4.85% |
| 12 | `Pepper__bell___Bacterial_spot` | Pepper | Bacterial Spot (*Xanthomonas campestris*) | 997 | 4.83% |
| 13 | `Tomato_Leaf_Mold` | Tomato | Leaf Mold (*Passalora fulva*) | 952 | 4.61% |
| 14 | `Tomato__Tomato_mosaic_virus` | Tomato | Tomato Mosaic Virus (ToMV) | 373 | 1.81% |
| 15 | `Potato___healthy` | Potato | Healthy Foliage (Baseline) | 152 | 0.74% |
| — | **Total** | **3 Species** | **15 Classes** | **20,638** | **100.00%** |

### Species-Level Aggregation
- **Tomato:** 16,011 images (77.58%) across 10 classes
- **Pepper:** 2,475 images (11.99%) across 2 classes
- **Potato:** 2,152 images (10.43%) across 3 classes

### Imbalance Ratio Analysis
$$\text{Class Imbalance Ratio} = \frac{N_{\max}}{N_{\min}} = \frac{3,208}{152} = \mathbf{21.11:1}$$

Because the largest class (`Tomato__Tomato_YellowLeaf__Curl_Virus`) is more than 21 times larger than the smallest class (`Potato___healthy`), models evaluated solely on standard Accuracy could achieve $>80\%$ while completely misclassifying minority classes. This demands **stratified dataset splitting** and **macro-averaged evaluation metrics**.

---

## 4. Features & Target Variables

### Target Variable ($y$)
A discrete categorical label corresponding to one of the 15 verified class folders (e.g. `Tomato_healthy`, `Potato___Early_blight`, `Pepper__bell___Bacterial_spot`). Target classes are encoded into integer IDs $[0, 14]$ during training and dynamically mapped back to human-readable strings during inference.

### Input Features ($X$)
Traditional Computer Vision numerical vectors extracted deterministically from preprocessed leaf photographs.

---

## 5. Image Preprocessing Methodology

Image preprocessing standardizes raw heterogeneous camera inputs prior to numerical feature extraction:

1. **Format Validation & Ingestion:** Loaded via Pillow and OpenCV, strictly validated against supported extensions (`.jpg`, `.jpeg`, `.png`).
2. **RGB Color Space Standardization:** Grayscale, palette, and RGBA images are converted to uniform 3-channel RGB (`uint8`).
3. **Spatial Resizing:** Resampled to a consistent spatial grid of $128 \times 128$ pixels using bilinear interpolation.
4. **Array Bound Assertions:** Ensures the array is shaped `(128, 128, 3)`, dtype is `uint8`, pixel intensities fall strictly within $[0, 255]$, and no NaN or Inf values exist.
5. **Exception Handling:** Unreadable or truncated files trigger a custom `ImagePreprocessingError` with descriptive logging.

### Critical Methodological Distinction: Resizing vs Normalization vs Scaling

| Concept | Operation Level | Mathematical Range | Purpose in Traditional CV |
| :--- | :--- | :--- | :--- |
| **Image Resizing** | Spatial pixel grid | $(128, 128, 3)$ coordinates | Standardizes spatial dimensions so cell/block grids (HOG, GLCM) operate on identical pixel geometries across all samples. |
| **Pixel Intensity Normalization** | Raw image matrix | Rescaling $[0, 255] \rightarrow [0.0, 1.0]$ | **Not applied to raw images.** Traditional descriptors (GLCM, color histograms) require discrete integer intensity levels $[0, 255]$ for discrete quantization binning. |
| **Feature Scaling** | Extracted numerical vector | $\mu = 0, \sigma = 1$ across dataset | **Applied inside Scikit-learn Pipeline via `StandardScaler`**. Standardizes features across the training set to prevent high-magnitude features from dominating distance metrics in SVM. |

---

## 6. Feature Engineering & Representation Ablation

### Handcrafted Visual Descriptors

#### A. Color Features (48 Dimensions)
- **RGB Color Histogram (24D):** Extracts 8 normalized bins per color channel ($R, G, B$) across range $[0, 256]$. Quantifies overall red/green/blue balance.
- **HSV Color Histogram (24D):** Converts RGB to HSV. Extracts 8 bins for Hue ($[0, 180]$), 8 for Saturation ($[0, 256]$), and 8 for Value ($[0, 256]$). 
  - *Botanical Rationale:* Hue cleanly decouples spectral chromaticity from lighting fluctuations. Healthy foliage exhibits a narrow green hue peak ($\approx 35^\circ - 85^\circ$), whereas fungal and viral infections cause pronounced shifts toward yellow chlorosis bands and dark brown/black necrotic peaks.

#### B. Texture Features (34 Dimensions)
- **Local Binary Patterns (LBP, 10D):** Evaluates circular neighbor intensity differences ($P = 8, R = 1$) with the `"uniform"` pattern formulation, generating a normalized 10-bin histogram. Captures leaf surface roughness and lesion perimeter texture.
- **Gray-Level Co-occurrence Matrix (GLCM, 24D):** Quantizes grayscale intensities to 32 discrete levels and calculates Haralick second-order spatial joint probabilities across distances $d \in \{1, 2\}$ and 4 angular directions $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$. Computes mean and standard deviation for 6 statistical properties:
  1. *Contrast:* Local intensity variation (high in necrotic spot boundaries).
  2. *Dissimilarity:* Absolute gray-level difference between pixel pairs.
  3. *Homogeneity:* Closeness of element distribution to GLCM diagonal.
  4. *Energy:* Textural uniformity (square root of ASM).
  5. *Correlation:* Linear dependence of gray levels across pixel offsets.
  6. *Angular Second Moment (ASM):* Measure of orderliness and homogeneity.

#### C. Shape Features (Diagnostic & Limitation Analysis)
- Analyzes leaf geometry via Otsu global thresholding after Gaussian filtering: Area, Perimeter, Circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$), and Aspect Ratio.
- **Documented Methodological Limitation:** In the PlantVillage collection, leaves are photographed on neutral laboratory backdrops with cast shadows, petiole cuts, and overlapping leaflets. Unsupervised segmentation frequently misclassifies shadows as leaf margins or fragments curled leaf blades. Because shape features from uncurated segmentation introduce severe noise rather than true botanical morphology, they are **deliberately excluded from the classification vector**.

#### D. Histogram of Oriented Gradients (HOG, 8,100 Dimensions)
- Captures localized edge orientations and structural gradient magnitudes.
- Parameters: $128 \times 128$ grayscale image, 9 orientation bins, pixels per cell = $(8, 8)$, cells per block = $(2, 2)$, block norm = `"L2-Hys"`.
- Spatial dimension derivation:
  $$\text{Blocks per axis} = \frac{128}{8} - 2 + 1 = 15 \implies 15 \times 15 = 225 \text{ total blocks}$$
  $$\text{Values per block} = 2 \times 2 \times 9 = 36 \implies 225 \times 36 = \mathbf{8,100} \text{ features}$$

---

### Empirical Feature Ablation Benchmark Results

To identify the most effective and computationally sound feature representation, an empirical ablation benchmark was conducted across all 9 descriptor subsets on a stratified 4,000-sample benchmark split:

| Feature Representation | Feature Dimension | Test Accuracy | Macro F1-Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Color + Texture (Selected Final)** | **82** | **86.50%** | **78.75%** | **0.13 s** |
| Color (RGB + HSV) | 48 | 84.17% | 76.92% | 0.09 s |
| HSV Histogram | 24 | 82.67% | 75.59% | 0.08 s |
| RGB Histogram | 24 | 72.00% | 63.28% | 0.10 s |
| Color + Texture + HOG | 8,182 | 66.50% | 53.92% | 40.80 s |
| HOG Gradients Only | 8,100 | 59.67% | 47.92% | 36.65 s |
| Texture (LBP + GLCM) | 34 | 59.33% | 45.21% | 0.09 s |
| GLCM Haralick Properties | 24 | 53.83% | 38.97% | 0.10 s |
| LBP Texture Only | 10 | 45.33% | 34.22% | 0.11 s |

#### Key Scientific Finding: The Curse of Dimensionality in Traditional Vision
Adding 8,100 HOG gradient features to Color + Texture increased the dimensionality by $100\times$ (from 82 to 8,182) but caused Macro F1 to drop by **24.8%** (from 78.75% down to 53.92%), while slowing training down by over **300-fold** (40.8 seconds vs 0.13 seconds). 

In traditional kernel classification with 15 classes, ultra-high dimensional spaces cause severe hyperspace sparsity, making it harder for the RBF kernel to establish clean hyperplane separation boundaries. Therefore, **`color_texture` (82 dimensions)** was selected as the optimal feature representation for production model training, while full modular support for `color_texture_hog` (8,182D) is maintained in `src/feature_extraction.py`.

---

## 7. Traditional Machine Learning Algorithms

### 1. Support Vector Machine (SVM)
- **Classifier:** `sklearn.svm.SVC`
- **Kernel Function:** Radial Basis Function (RBF):
  $$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
- **Mathematical Role of $C$ ($C = 5.0$):** Controls the tradeoff between maximizing the decision boundary margin and penalizing classification slack variables $\xi_i$. Larger $C$ creates tighter margins and prioritizes classifying training instances correctly.
- **Mathematical Role of $\gamma$ ($\gamma = \text{"scale"}$):** Determines the reach of individual training samples. Setting $\gamma = \frac{1}{D \cdot \text{Var}(X)}$ adapts the Gaussian radius to the variance of the 82 feature dimensions.
- **Multi-Class Strategy:** One-vs-One (OvO) decision framework with probability calibration via Platt scaling (`probability=True`).

### 2. Random Forest Classifier
- **Classifier:** `sklearn.ensemble.RandomForestClassifier`
- **Architecture:** Ensemble of $B = 200$ decorrelated classification decision trees.
- **Bagging (Bootstrap Aggregation):** Each tree is trained on a bootstrap sample of $N$ instances drawn with replacement from the training set, minimizing model variance without increasing bias.
- **Random Subspace Projection:** At each node split, evaluates a random subset of $m = \sqrt{D} \approx 9$ features to decorrelate individual trees.
- **Handling Class Imbalance:** `class_weight="balanced"` weights node impurities inversely proportional to class frequencies:
  $$w_c = \frac{N}{K \cdot N_c}$$

---

## 8. Machine Learning Pipelines & Data Leakage Prevention

To guarantee strict academic integrity and zero data leakage:

```text
Original Leaf Image
       │
       ▼
Image Preprocessing (128x128 RGB)
       │
       ▼
Feature Extraction (Color + Texture, 82D)
       │
       ▼
Train-Test Split (80% Train, 20% Test, Stratified)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
   SVM Pipeline                          Random Forest Pipeline
   ├── StandardScaler                    └── RandomForestClassifier
   └── SVC (RBF Kernel, C=5.0)                (n_estimators=200, balanced)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
               Predicted Disease Class [0-14]
```

### Leakage Prevention Rules Followed
1. **Never scale before splitting:** `StandardScaler` is **never** fitted on the full dataset. It is strictly embedded as step 1 in the Scikit-learn Pipeline.
2. **Strict split isolation:** Scaler parameters (mean $\mu$ and standard deviation $\sigma$) are learned strictly from the 16,510 training instances (`fit_transform`) and only used to normalize the 4,128 test instances (`transform`).
3. **Cross-Validation safety:** During hyperparameter optimization, the Pipeline ensures the scaler is re-fitted inside each cross-validation training fold without leaking validation fold data.

---

## 9. Dataset Splitting & Stratification

- **Training Split:** 80% ($N = 16,510$ images)
- **Testing Split:** 20% ($N = 4,128$ images)
- **Stratification:** Mandatory (`stratify=y`) to maintain the identical 21.11:1 class ratio across both partitions.
- **Reproducibility Seed:** `random_state=42`.

---

## 10. Feature Scaling

Feature scaling standardizes each numerical column to zero mean and unit variance:
$$z = \frac{x - \mu}{\sigma}$$

- **Mandatory for SVM:** The RBF kernel measures Euclidean distances $\|x - x'\|^2$. Without scaling, unnormalized GLCM properties or histogram bins with wider variances would dominate the optimization objective.
- **Scale-Invariance of Random Forest:** Decision trees perform greedy monotonic threshold splits along individual orthogonal feature axes ($x_j \le \theta$). Consequently, Random Forest does not require feature scaling.

---

## 11. Hyperparameter Optimization

Hyperparameters were systematically explored using 3-fold `StratifiedKFold` cross-validation on training data ($N = 16,510$), optimizing for **Macro F1-score**:

### SVM Optimization Grid
- $C \in [0.1, 1.0, 5.0, 10.0]$
- $\gamma \in [\text{"scale"}, \text{"auto"}, 0.01, 0.1]$
- **Optimal Parameters:** `C=5.0`, `gamma="scale"`, `kernel="rbf"`. Cross-validation Macro F1: **0.9412**.

### Random Forest Optimization Grid
- `n_estimators` $\in [100, 200, 300]$
- `max_depth` $\in [15, 25, \text{None}]$
- `min_samples_split` $\in [2, 5]$
- `class_weight` $\in [\text{"balanced"}, \text{None}]$
- **Optimal Parameters:** `n_estimators=200`, `max_depth=25`, `class_weight="balanced"`. Cross-validation Macro F1: **0.9008**.

---

## 12. Evaluation Metrics

Because of the severe 21.11:1 class imbalance, overall Accuracy alone is insufficient. We compute:

1. **Accuracy:** Fraction of all test samples correctly predicted:
   $$\text{Accuracy} = \frac{\sum_{i=1}^K TP_i}{N}$$
2. **Macro Precision:** Unweighted arithmetic mean of Precision across all 15 classes:
   $$\text{Macro Precision} = \frac{1}{K}\sum_{i=1}^K \frac{TP_i}{TP_i + FP_i}$$
3. **Macro Recall (Sensitivity):** Unweighted arithmetic mean of Recall across all 15 classes:
   $$\text{Macro Recall} = \frac{1}{K}\sum_{i=1}^K \frac{TP_i}{TP_i + FN_i}$$
4. **Macro F1-Score (Primary Benchmark):** Harmonic mean of Macro Precision and Macro Recall:
   $$\text{Macro F1} = 2 \cdot \frac{\text{Macro Precision} \cdot \text{Macro Recall}}{\text{Macro Precision} + \text{Macro Recall}}$$
5. **Weighted F1-Score:** F1-score weighted by the true support (sample count) of each class.
6. **Confusion Matrix ($15 \times 15$):** Full matrix tracking all true positives, false positives, and false negatives.

---

## 13. Experimental Results & Model Comparison

Both final pipelines were evaluated on the held-out test split of **4,128 images**. All figures below are actual, verified empirical results:

### Side-by-Side Model Comparison Table

| Evaluation Metric | Support Vector Machine (SVM) | Random Forest Classifier | Performance Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **95.37%** | **91.23%** | **+4.14% (SVM)** |
| **Macro Precision** | **95.06%** | **90.21%** | **+4.85% (SVM)** |
| **Macro Recall** | **93.99%** | **90.41%** | **+3.58% (SVM)** |
| **Macro F1-Score** | **94.44%** | **90.25%** | **+4.19% (SVM)** |
| **Weighted Precision** | **95.36%** | **91.26%** | **+4.10% (SVM)** |
| **Weighted Recall** | **95.37%** | **91.23%** | **+4.14% (SVM)** |
| **Weighted F1-Score** | **95.35%** | **91.22%** | **+4.13% (SVM)** |
| **Training Execution Time** | **12.45 seconds** | **18.20 seconds** | **1.46x Faster (SVM)** |
| **Inference Latency** | **~2.1 ms / image** | **~3.4 ms / image** | **SVM Faster** |

### Per-Class Test Performance (SVM Classifier)

| Class Name | Precision | Recall | F1-Score | Support ($N_{test}$) |
| :--- | :---: | :---: | :---: | :---: |
| `Pepper__bell___Bacterial_spot` | 0.9415 | 0.9698 | 0.9554 | 199 |
| `Pepper__bell___healthy` | 0.9831 | 0.9831 | 0.9831 | 296 |
| `Potato___Early_blight` | 0.9406 | 0.9500 | 0.9453 | 200 |
| `Potato___Late_blight` | 0.9029 | 0.9300 | 0.9163 | 200 |
| `Potato___healthy` | 0.8667 | 0.8667 | 0.8667 | 30 |
| `Tomato_Bacterial_spot` | 0.9329 | 0.9435 | 0.9382 | 425 |
| `Tomato_Early_blight` | 0.8980 | 0.8800 | 0.8889 | 200 |
| `Tomato_Late_blight` | 0.9456 | 0.9503 | 0.9479 | 382 |
| `Tomato_Leaf_Mold` | 0.9628 | 0.9421 | 0.9524 | 190 |
| `Tomato_Septoria_leaf_spot` | 0.9231 | 0.9153 | 0.9192 | 354 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 0.9472 | 0.9104 | 0.9285 | 335 |
| `Tomato__Target_Spot` | 0.9247 | 0.8897 | 0.9069 | 281 |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 0.9938 | 0.9953 | 0.9945 | 642 |
| `Tomato__Tomato_mosaic_virus` | 0.9863 | 0.9600 | 0.9730 | 75 |
| `Tomato_healthy` | 0.9780 | 0.9748 | 0.9764 | 318 |
| **Macro Average** | **0.9506** | **0.9399** | **0.9444** | **4,128** |
| **Weighted Average** | **0.9536** | **0.9537** | **0.9535** | **4,128** |

### Best-Performing Model Determination
**Support Vector Machine (SVM) with RBF Kernel ($C=5.0$, `gamma='scale'`) is definitively the superior algorithm**, outperforming Random Forest across every metric (+4.19% Macro F1, +4.14% Accuracy). 

SVM constructs optimal maximum-margin hyperplanes in Hilbert space, excelling on dense, continuous numerical features (normalized color histograms and Haralick co-occurrences), whereas axis-aligned decision trees struggle with subtle diagonal correlations in color-texture feature spaces.

---

## 14. Project Architecture

The project is structured according to professional software and machine learning engineering standards:

```text
plant-disease-ml/
│
├── data/
│   ├── raw/
│   │   └── PlantVillage/              # Original raw image dataset (15 classes, 20,638 images)
│   │
│   └── processed/
│       ├── metadata.csv               # Scanned tabular metadata catalog (20,638 rows)
│       ├── features.npy               # Precomputed 82D feature array (20638, 82)
│       ├── labels.npy                 # Integer label array (20638,)
│       └── feature_dataset.npz        # Unified compressed NPZ archive
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Dataset scanning, statistics, imbalance, and leaf visualization
│   ├── 02_image_preprocessing.ipynb   # Resizing vs normalization vs scaling, channel decomposition
│   ├── 03_feature_engineering.ipynb   # Color (RGB+HSV), Texture (LBP+GLCM), HOG, and feature ablation
│   ├── 04_model_training.ipynb        # Leak-free pipelines, cross-validation, and Joblib serialization
│   └── 05_model_evaluation.ipynb      # Confusion matrices, per-class metrics, model comparison & inference
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Global directory paths, image constants, and parameters
│   ├── data_loader.py                 # Dataset scanning, statistics, integrity check, and stratified split
│   ├── preprocessing.py               # Image loading, RGB conversion, 128x128 resizing, validation
│   ├── feature_extraction.py          # RGB, HSV, LBP, GLCM, Shape diagnostics, and HOG extractors
│   ├── feature_dataset.py             # Parallel multi-core feature extraction and serialization
│   ├── feature_comparison.py          # Empirical feature ablation benchmarking engine
│   ├── pipeline.py                    # Scikit-learn Pipeline factories for SVM and Random Forest
│   ├── optimize.py                    # Hyperparameter tuning with StratifiedKFold and Macro F1
│   ├── train.py                       # Full training orchestrator, Joblib persistence, metadata registry
│   ├── evaluate.py                    # Metric computation, confusion matrix heatmaps, report exports
│   └── predict.py                     # Production inference engine with confidence and error handling
│
├── app/
│   └── app.py                         # Interactive Streamlit Web Application
│
├── models/
│   ├── svm_pipeline.joblib            # Trained production SVM Pipeline (StandardScaler + SVC)
│   ├── random_forest_pipeline.joblib  # Trained production Random Forest Pipeline
│   └── model_metadata.json            # Model parameters, classes, timestamps, and test metrics
│
├── results/
│   ├── figures/
│   │   ├── feature_comparison.png     # Macro F1 vs descriptor representation bar chart
│   │   ├── model_comparison.png       # Side-by-side SVM vs Random Forest performance bar chart
│   │   ├── svm_confusion_matrix.png   # Normalized 15x15 SVM confusion matrix heatmap
│   │   ├── svm_per_class_metrics.png  # Per-class Precision, Recall, and F1 for SVM
│   │   ├── random_forest_confusion_matrix.png
│   │   └── random_forest_per_class_metrics.png
│   │
│   ├── metrics/
│   │   ├── svm_metrics.json           # Serialized test scores for SVM
│   │   ├── svm_classification_report.txt
│   │   ├── random_forest_metrics.json
│   │   └── random_forest_classification_report.txt
│   │
│   ├── model_comparison.csv           # Tabular comparison summary (Accuracy, Macro F1, etc.)
│   ├── feature_comparison.csv         # Tabular feature ablation results (9 descriptor sets)
│   └── evaluation_report.txt          # Comprehensive textual evaluation audit report
│
├── tests/
│   ├── test_preprocessing.py          # Array bounds, NaN/Inf checks, format rejection
│   ├── test_features.py               # Feature dimension, non-negativity, and histogram sums
│   ├── test_pipeline.py               # Leak-free scaling, pipeline structure, serialization roundtrips
│   └── test_prediction.py             # Single-image inference, confidence, error interception
│
├── colab/
│   └── plant_disease_classification.ipynb # Standalone end-to-end Google Colab notebook (CPU runtime)
│
├── scripts/
│   └── build_all_notebooks.py         # Automated notebook build script
│
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Dependency lockfile
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Git exclusion rules
```

---

## 15. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Linux, macOS, or Windows
- Git

### Step-by-Step Installation

```bash
# 1. Clone repository and navigate to folder
git clone <repository_url>
cd plant-disease-ml

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows PowerShell

# 3. Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 16. Training Instructions

To re-run the complete model training, hyperparameter optimization, and evaluation workflow:

```bash
# Run baseline training with default optimal Color + Texture representation:
python -m src.train

# Run with hyperparameter tuning enabled:
python -m src.train --optimize

# Run feature representation ablation benchmarking:
python -m src.feature_comparison
```

This outputs:
- Serialized pipelines: `models/svm_pipeline.joblib` and `models/random_forest_pipeline.joblib`
- Registry update: `models/model_metadata.json`
- Metrics & plots: `results/model_comparison.csv`, `results/figures/`, `results/metrics/`

---

## 17. Streamlit Web Application

An interactive web application is provided for demonstration, testing, and university presentation.

### Starting the Web App
```bash
streamlit run app/app.py
```

The application opens automatically in your web browser at `http://localhost:8501`.

### Web Application Features
- **Leaf Input:** Upload your own JPEG/PNG leaf photo, or choose from 15 built-in benchmark samples.
- **Model Selector:** Instantly toggle between **Support Vector Machine (RBF Kernel)** and **Random Forest (Ensemble)**.
- **Diagnostic Result Card:** Displays predicted plant species, disease pathology, confidence score, and top candidate probabilities with progress bars.
- **Traditional Feature Explanation:** Detailed breakdown of how the 82 visual descriptors were computed.
- **Agricultural Disclaimer:** Field guidance and diagnostic limitation notice.

---

## 18. Prediction Instructions

### CLI Prediction (Terminal)
Classify any leaf image directly from the command line:

```bash
# Predict using SVM (default):
python src/predict.py "data/raw/PlantVillage/Tomato_healthy/0a0de248-1502-4017-802c-ae66750058b8___GH_HL Leaf 259.1.JPG" --model svm

# Predict using Random Forest:
python src/predict.py "data/raw/PlantVillage/Potato___Early_blight/0a8a6869-6dcb-4db0-a152-78d1ef3967b2___RS_Early.B 8470.JPG" --model rf
```

### Python API Usage
Integrate the predictor directly into your Python scripts:

```python
from src.predict import PlantDiseasePredictor

# Initialize predictor (loads pipeline and registered class metadata)
predictor = PlantDiseasePredictor(model_type="svm")

# Predict on an image path or PIL Image object
result = predictor.predict("path/to/leaf_image.jpg")

print(f"Species:    {result['species']}")
print(f"Disease:    {result['disease']}")
print(f"Class:      {result['class_name']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
print("Top 3 Candidates:")
for cls, prob in result["top_probabilities"].items():
    print(f"  - {cls}: {prob * 100:.2f}%")
```

---

## 19. Jupyter Notebooks Guide

The `notebooks/` directory contains five standalone, fully-executed, educational notebooks:

1. **`01_dataset_exploration.ipynb`:** Scans `data/raw/PlantVillage/`, computes exact statistics (20,638 images, 15 classes, 3 species), calculates class imbalance ratios (21.11:1), and visualizes sample leaves across all classes.
2. **`02_image_preprocessing.ipynb`:** Explains the theoretical distinction between image resizing, pixel normalization, and feature scaling. Demonstrates $128 \times 128$ RGB resizing, channel decomposition, and input validation.
3. **`03_feature_engineering.ipynb`:** Implements RGB and HSV color histograms, LBP texture, GLCM Haralick properties, diagnostic shape analysis, and HOG. Executes the 9-group feature ablation experiment.
4. **`04_model_training.ipynb`:** Demonstrates data leakage prevention, constructs Scikit-learn pipelines (`StandardScaler` + `SVC`, `RandomForestClassifier`), conducts cross-validated tuning, and serializes models with Joblib.
5. **`05_model_evaluation.ipynb`:** Computes all classification metrics on the 4,128-sample held-out test split, plots normalized confusion matrices, compares SVM vs Random Forest, and demonstrates inference on raw leaf images.

To launch Jupyter Lab:
```bash
jupyter lab notebooks/
```

---

## 20. Google Colab Workflow

A standalone Google Colab notebook is available at `colab/plant_disease_classification.ipynb`.

### Execution Steps
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `colab/plant_disease_classification.ipynb`.
3. Select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ CPU** (No GPU required).
4. Run all cells sequentially. The notebook:
   - Mounts Google Drive (optional).
   - Installs required dependencies.
   - Sets up preprocessing and feature extraction functions.
   - Trains and optimizes SVM and Random Forest pipelines.
   - Displays confusion matrices and performance tables.
   - Provides an interactive file upload widget for predicting custom leaf images.

---

## 21. Automated Testing & Quality Assurance

All core modules are verified by an automated test suite implemented in `pytest`:

```bash
pytest tests/ -v
```

### Test Coverage Highlights
- **`tests/test_preprocessing.py` (8 tests):** Validates array loading, RGB shape assertions `(128, 128, 3)`, dtype `uint8`, pixel ranges $[0, 255]$, NaN/Inf detection, and format rejection.
- **`tests/test_features.py` (9 tests):** Asserts exact feature dimensionalities (RGB: 24, HSV: 24, LBP: 10, GLCM: 24, HOG: 8,100), histogram sum normalization, non-negativity, and shape diagnostics.
- **`tests/test_pipeline.py` (5 tests):** Validates pipeline structure, verifies that `StandardScaler` calculates parameters strictly on training samples (leakage prevention), and ensures Joblib serialization round-trip consistency.
- **`tests/test_prediction.py` (3 tests):** Validates the `PlantDiseasePredictor` output schema, confidence score bounds $[0.0, 1.0]$, top-3 probability formatting, and rejection of unreadable inputs.

**Results:** **25 passed in 1.34s** with 100% success rate.

---

## 22. Model Limitations & Threat to Validity

When evaluating this traditional machine learning system, the following operational limitations must be noted:

1. **Laboratory Backdrop Bias:** PlantVillage images were photographed on uniform laboratory cardboard and neutral backdrops. Field images taken in uncontrolled farming environments with soil, sunlight glare, weeds, or shadows may suffer performance degradation.
2. **Closed-World Assumption:** The model is trained on 15 specific classes across Pepper, Potato, and Tomato. It cannot diagnose pathogens, weeds, or plant species outside this closed vocabulary.
3. **Co-occurring Infections & Abiotic Stresses:** Foliar pathogens frequently co-occur in the field (e.g. early blight alongside spider mite infestation) or mimic nitrogen chlorosis. The system assumes a single primary pathology per leaf image.
4. **Probability Calibration Notice:** While SVM probabilities are derived via Platt scaling (`probability=True`), they represent posterior estimates over the closed training distribution and are not an absolute agricultural guarantee.

> **Agricultural Notice:** This software is intended for educational, academic, and decision-support research. Automated predictions should not replace professional inspection by certified agronomists or laboratory phytopathological assays.

---

## 23. Future Improvements

1. **Automated Foreground Segmentation:** Implementing background removal (e.g., GrabCut or U-Net foreground masking) prior to feature extraction to support in-the-wild field leaf photographs.
2. **Multi-Label Pathology Support:** Extending the OvO SVM classification architecture to multi-label ranking for leaves suffering from simultaneous fungal and pest damage.
3. **Edge Deployment:** Quantizing and compiling the 82-dimensional feature extraction pipeline for offline mobile execution on microcontroller/Raspberry Pi hardware in rural agricultural communities.
4. **Active Learning Expansion:** Integrating an active learning feedback loop in the Streamlit application to allow expert agronomists to flag uncertain predictions and expand the training set.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
