# Project Context: Intelligent Network Intrusion Detection System (NIDS)

## 1. Project Title & Objective

- **Title**: Intelligent Network Intrusion Detection System (NIDS) Using Deep Learning
- **Objective**: Develop an intelligent Network Intrusion Detection System capable of detecting and classifying multi-class malicious network traffic on the **CICIDS2017** dataset. The project investigates data quality, statistical feature selection (ANOVA), CNN-GRU sequence modeling, conventional classification performance, zero-day generalization via Leave-One-Attack-Out (LOAO) evaluation, and protocol/data-leakage auditing.

---

## 2. Dataset Overview: CICIDS2017

The **CICIDS2017** dataset contains benign network traffic and real-world network attacks captured over 5 consecutive days. For this project, seven target classes were selected for multi-class intrusion detection:

- `0`: **BENIGN** (Normal network traffic)
- `1`: **DoS Hulk**
- `2`: **DDoS**
- `3`: **PortScan**
- `4`: **DoS GoldenEye**
- `5`: **FTP-Patator**
- `6`: **SSH-Patator**

---

## 3. Data Preprocessing & Feature Engineering Pipeline

### Data Cleaning & Quality Analysis
1. **Missing & Infinite Value Handling**: Replaced or dropped invalid `NaN` and `Infinity` numerical values across raw CSV files.
2. **Anomalous Row Filtering**: Removed 35 rows containing invalid negative feature values.
3. **Conflict Detection & Removal**: Identified and removed duplicate feature vectors with conflicting labels at both local file and global dataset levels.
4. **Constant Feature Removal (8 features)**:
   - `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`
5. **Exact Duplicate Feature Removal (9 features)**:
   - `Subflow Fwd Packets`, `Subflow Bwd Packets`, `Fwd Header Length.1`, `CWE Flag Count`, `SYN Flag Count`, `Avg Fwd Segment Size`, `Avg Bwd Segment Size`, `Subflow Fwd Bytes`, `Subflow Bwd Bytes`
6. **Final Feature-Cleaned Dataset**:
   - Total Rows: **2,805,715**
   - Total Usable Features: **61** (out of original 78 features)

---

## 4. Train / Test Split & Feature Selection

### Stratified 80/20 Split
- **Training Set**: 2,244,572 rows (80%)
- **Testing Set**: 561,143 rows (20%)
- Class proportions are strictly preserved between training and testing splits.

### ANOVA F-Test Feature Selection
ANOVA F-test ranking was computed on the training set across all 61 usable features. Feature subsets were constructed for comparison:
- **Top 10**, **Top 20**, **Top 30**, **Top 40**, and **Top 61** features.

**Top 10 ANOVA Ranked Features**:
1. `Bwd Packet Length Std`
2. `Bwd Packet Length Mean`
3. `Bwd Packet Length Max`
4. `Packet Length Std`
5. `Fwd IAT Std`
6. `Max Packet Length`
7. `Packet Length Variance`
8. `Idle Min`
9. `Idle Mean`
10. `Average Packet Size`

---

## 5. Scaling & Sequence Framing

### Feature Scaling
- `StandardScaler` is fitted **strictly on the training set** for each feature subset and then applied to transform both training and testing datasets, preventing data leakage.

### Sequence Framing (3D Input Structure)
- **Window Size (Sequence Length)**: 10 flows
- **Stride**: 10
- Converts 2D tabular features into 3D sequence tensors of shape `(Batch_Size, 10, Num_Features)`.
- Class-pure sequence framing (`src/create_class_sequences.py`) structures sequence windows to ensure consistent intra-sequence labeling.

---

## 6. CNN-GRU Model Architecture & Training

### Neural Network Architecture
- **Input Layer**: `(10, Num_Features)`
- **Conv1D Layer 1**: 64 filters, kernel size = 3, activation = ReLU, padding = 'same'
- **Batch Normalization**
- **MaxPooling1D**: pool size = 2
- **Conv1D Layer 2**: 128 filters, kernel size = 3, activation = ReLU, padding = 'same'
- **Batch Normalization**
- **GRU Layer**: 64 units (`return_sequences=False`)
- **Dropout**: 0.3
- **Dense Layer**: 64 units, activation = ReLU
- **Dropout**: 0.3
- **Output Dense Layer**: 7 units, Softmax activation

### Training Setup
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Class Weights**: Inverse frequency weighting (`data/processed/class_weights/class_weights.csv`) to handle class imbalance.
- **Callbacks**:
  - `EarlyStopping` (patience = 5, restore_best_weights = True)
  - `ModelCheckpoint` (saves best `.keras` model based on validation loss)
  - `ReduceLROnPlateau` (factor = 0.5, patience = 2, min_lr = 1e-6)

---

## 7. Performance & Research Findings

### Conventional Test Results (Stratified 80/20 Test Set)
| Feature Set | Features | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | 0.9995 | 0.9986 | 0.9931 | 0.9995 |
| **Top-20** | 20 | 1.0000 | 0.9991 | 0.9985 | 1.0000 |
| **Top-30** | 30 | 1.0000 | 0.9991 | 0.9989 | 1.0000 |
| **Top-40** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Zero-Day Generalization (LOAO Experiment Results)
The **Leave-One-Attack-Out (LOAO)** experiment evaluates model performance when an entire attack family is excluded from training and tested as an unseen zero-day attack:

| Unseen Attack Class | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **DoS Hulk** | 91.20% | 100.00% | 2.44% | 4.77% |
| **DDoS** | 94.96% | 99.30% | 5.51% | 10.44% |
| **PortScan** | 93.89% | 100.00% | 6.02% | 11.36% |
| **DoS GoldenEye** | 99.61% | 100.00% | 13.17% | 23.28% |
| **FTP-Patator** | 99.66% | 66.67% | 2.53% | 4.88% |
| **SSH-Patator** | 99.76% | 100.00% | 7.69% | 14.29% |
| **MEAN** | — | — | **6.23%** | **11.50%** |

- **Key Takeaway**: Despite achieving near ~100% accuracy on standard stratified test splits, the model exhibits very low recall (mean 6.23%) on unseen attack families. This demonstrates that standard test splits mask zero-day generalization weaknesses.

### Protocol & Data Leakage Findings
- **Feature Vector Overlap**: Deep leakage audit (`src/deep_leakage_check.py`) revealed that **21.23% of training rows** and **25.90% of testing rows** share identical feature vectors.
- **Label Mismatches**: 81 unique feature patterns exist with conflicting labels across train and test sets.
- **Implication**: Overlapping network flow signatures in standard random/stratified splits artificially inflate test metrics, reinforcing why LOAO testing is essential for realistic evaluation.

### What Results to Present (Paper & Presentation)
1. **Feature Subset Comparison**: Top-10 to Top-40 progression shows how statistical selection reduces feature dimensions while maintaining high classification capacity on known traffic.
2. **LOAO Generalization Analysis**: Key research contribution highlighting the disparity between standard test metrics and true zero-day attack detection.
3. **Data Leakage & Protocol Critique**: Critical evaluation of CICIDS2017 evaluation protocols.

---

## 8. Presentation Demo Workflow & Commands

### Interactive Demo Setup
- **Demo Script**: `src/presentation_demo.py`
- **Model Used**: `models/cnn_gru_top40_final.keras`
- **Dataset Used**: `data/processed/sequences_class/test_top_40_X.npy` and `test_top_40_y.npy`

### Exact Command to Run Demo
```bash
python src/presentation_demo.py
```

### Seven Verified Demo Samples
Use these pre-verified sample indices during live demonstration:

| Sample Index | Ground-Truth Class | Verified Output |
| :---: | :--- | :--- |
| **Sample 0** | `BENIGN` (Normal) | Correctly classified as BENIGN |
| **Sample 3** | `DoS Hulk` | Correctly classified as DoS Hulk |
| **Sample 32** | `DDoS` | Correctly classified as DDoS |
| **Sample 9** | `PortScan` | Correctly classified as PortScan |
| **Sample 91** | `DoS GoldenEye` | Correctly classified as DoS GoldenEye |
| **Sample 151** | `FTP-Patator` | Correctly classified as FTP-Patator |
| **Sample 5040** | `SSH-Patator` | Correctly classified as SSH-Patator |

> **Note for Teammates**: These indices represent curated demonstration samples designed for presentation purposes. They demonstrate that the model successfully classifies examples from all seven classes, but do not imply 100% confidence across all arbitrary network traffic samples.

---

## 9. Important Scripts & Project Directory Structure

```text
PS26-PAPER1/
├── data/
│   ├── raw/                           [Local only: Raw CICIDS2017 CSV files]
│   └── processed/
│       ├── cicids2017_final.csv       [Local only: Master cleaned CSV]
│       ├── class_weights/
│       │   └── class_weights.csv      [Tracked: Calculated class weights]
│       ├── feature_selection/
│       │   ├── anova_feature_ranking.csv [Tracked: Ranked 61 features]
│       │   └── anova_datasets/        [Local only: Top-N feature CSVs]
│       ├── scaled/                    [Local only: Scaled train/test CSVs]
│       ├── sequences/                 [Local only: Sliding window .npy tensors]
│       └── sequences_class/           [Local only: Class-pure .npy tensors]
│
├── models/                            [Local only / Git LFS: Trained .keras model files]
│   ├── cnn_gru_top10_final.keras
│   ├── cnn_gru_top20_final.keras
│   ├── cnn_gru_top30_final.keras
│   ├── cnn_gru_top40_final.keras      [Model used for presentation demo]
│   └── loao/                          [LOAO model checkpoints]
│
├── notebooks/                         [Tracked: EDA & feature analysis notebooks]
│   ├── 01_data_analysis.ipynb
│   └── 02_feature_analysis.ipynb
│
├── results/                           [Tracked: Performance reports & plots]
│   ├── cnn_gru_top10_report.txt
│   ├── cnn_gru_top20_report.txt
│   ├── cnn_gru_top30_report.txt
│   ├── cnn_gru_top40_report.txt
│   ├── final/                         [Summary CSVs and figures for paper]
│   ├── loao/                          [LOAO evaluation reports and plots]
│   └── protocol_check/                [Leakage audit reports and logs]
│
├── src/                               [Tracked: All pipeline and evaluation Python scripts]
│   ├── preprocessing.py              # Raw CSV cleaning & NaN/Inf handling
│   ├── feature_cleaning.py           # Constant/duplicate feature removal
│   ├── split_dataset.py              # Stratified 80/20 train/test split
│   ├── anova_selection.py            # ANOVA F-test feature selection ranking
│   ├── create_anova_datasets.py      # Sub-dataset generation for Top-N features
│   ├── scale_dataset.py              # StandardScaler application
│   ├── class_weights.py              # Class weight calculation
│   ├── create_sequences.py           # Sliding window sequence tensor generation
│   ├── create_class_sequences.py     # Class-pure sequence tensor generation
│   ├── train_cnn_gru.py              # CNN-GRU model training & evaluation
│   ├── presentation_demo.py          # Interactive CLI demo for Top-40 model
│   ├── demo_intrusion_detection.py   # Alternative demo script for Top-30 model
│   ├── find_demo_attacks.py          # Test sample lookup utility
│   ├── loao_experiment.py            # Leave-One-Attack-Out zero-day experiment
│   ├── strict_loao.py                # Strict zero-leakage LOAO implementation
│   ├── analyze_loao.py               # Aggregates LOAO recall and F1 metrics
│   ├── tsne_analysis.py              # t-SNE feature space visualization
│   ├── protocol_sanity_check.py      # Pipeline structure sanity verifier
│   ├── deep_leakage_check.py         # Train/test feature vector overlap auditor
│   ├── inspect_overlap_mismatch.py   # Label mismatch analyzer for duplicate flows
│   ├── final_results_table.py        # Generates comparison tables
│   ├── create_paper1_figures.py      # Generates paper/presentation plot figures
│   └── final_figure_inventory.py     # Figure file inventory checker
│
├── PROJECT_CONTEXT.md                 [Tracked: Project context & teammate guide]
├── README.md                          [Tracked: Repository summary]
├── requirements.txt                   [Tracked: Python dependencies]
└── .gitignore                         [Tracked: Git exclusion rules]
```

---

## 10. Troubleshooting & Teammate Guidelines

### Important Guidelines for Teammates
1. **Do NOT Delete or Overwrite Existing Models**: Pre-trained `.keras` models in `models/` represent verified experiment runs.
2. **Do NOT Regenerate Datasets unnecessarily**: Large processed CSV and `.npy` sequence files take substantial time and disk space to compute. Use existing `.npy` files for testing.
3. **Execution Context**: Always run scripts from the project root directory (`D:\PS26-PAPER1`).
4. **Environment Requirements**: Ensure python virtual environment has required packages installed (`tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).
5. **TensorFlow Warning on Windows**: TensorFlow >= 2.11 runs on CPU on Windows native. This is expected and CPU inference is fast enough for the interactive presentation demo.

---

## 11. Overall Project Workflow Summary

```text
Raw CICIDS2017 CSV Data (data/raw/)
       ↓
Preprocessing & Quality Cleaning (src/preprocessing.py)
       ↓
Feature Anomaly & Duplicate Removal (src/feature_cleaning.py)
       ↓
Stratified 80/20 Train/Test Split (src/split_dataset.py)
       ↓
ANOVA Statistical Feature Selection (src/anova_selection.py)
       ↓
StandardScaler Preprocessing (src/scale_dataset.py)
       ↓
Sequence Tensor Generation (src/create_class_sequences.py)
       ↓
CNN-GRU Model Training & Class Weighting (src/train_cnn_gru.py)
       ↓
Conventional Evaluation & LOAO Zero-Day Audit (src/loao_experiment.py)
       ↓
Protocol & Leakage Integrity Audit (src/deep_leakage_check.py)
       ↓
Live Presentation Demo (python src/presentation_demo.py)
```
