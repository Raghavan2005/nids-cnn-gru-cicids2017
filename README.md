# PS26 – Intelligent Network Intrusion Detection System

## Project Title

Intelligent Network Intrusion Detection System (NIDS) Using Deep Learning

## Project Objective

The objective of this project is to develop an intelligent Network Intrusion Detection System capable of detecting and classifying malicious network traffic using deep learning.

The current research work focuses on the CICIDS2017 dataset and investigates data quality, feature redundancy, statistical feature selection, and preparation of the dataset for a CNN-GRU based NIDS.

---

# Current Research Pipeline

The current preprocessing and feature-selection pipeline is:

```text
Raw CICIDS2017 Dataset
        ↓
Data Quality Analysis
        ↓
Conflict Detection and Removal
        ↓
NaN / Infinity Removal
        ↓
Class Selection
        ↓
Global Conflict Cleaning
        ↓
Feature Cleaning
        ↓
Train/Test Split
        ↓
ANOVA Feature Selection
        ↓
Feature Subset Creation
        ↓
Feature Scaling
        ↓
CNN-GRU Model
```

---

# Dataset

Dataset:
- CICIDS2017

Selected classes:
1. BENIGN
2. DoS Hulk
3. DDoS
4. PortScan
5. DoS GoldenEye
6. FTP-Patator
7. SSH-Patator

The original dataset contains multiple attack categories. For the current experiment, seven classes were selected.

---

# Data Cleaning

The preprocessing pipeline identified:
- Missing values
- Infinite values
- Duplicate rows
- Duplicate feature vectors
- Conflicting feature vectors with different labels
- Constant features
- Exact duplicate features
- Invalid negative feature values

### Conflict Cleaning

Feature vectors occurring with multiple labels were removed.
This was performed both at the file-level preprocessing stage and through a final global conflict-cleaning stage.

After global conflict cleaning:
- Conflicting feature groups remaining: 0
- Missing values: 0
- Infinite values: 0

---

# Feature Cleaning

The cleaned dataset initially contained:
- Rows: 2,805,750
- Columns: 79
- Features: 78

35 anomalous rows containing invalid negative values were removed.

8 constant features were removed:
- Bwd PSH Flags
- Bwd URG Flags
- Fwd Avg Bytes/Bulk
- Fwd Avg Packets/Bulk
- Fwd Avg Bulk Rate
- Bwd Avg Bytes/Bulk
- Bwd Avg Packets/Bulk
- Bwd Avg Bulk Rate

9 exact duplicate features were removed:
- Subflow Fwd Packets
- Subflow Bwd Packets
- Fwd Header Length.1
- CWE Flag Count
- SYN Flag Count
- Avg Fwd Segment Size
- Avg Bwd Segment Size
- Subflow Fwd Bytes
- Subflow Bwd Bytes

Final feature-cleaned dataset:
- Rows: 2,805,715
- Columns: 62
- Features: 61
- Missing values: 0
- Infinite values: 0

---

# Train/Test Split

An 80/20 stratified train/test split was performed.

Training:
- 2,244,572 rows

Testing:
- 561,143 rows

The class proportions were preserved between training and testing sets.

---

# ANOVA Feature Selection

ANOVA F-test was applied to the training data.

61 features were ranked according to their ANOVA F-score.

The following feature subsets were generated:
- Top 10
- Top 20
- Top 30
- Top 40
- Top 61

The top-ranked features include:
1. Bwd Packet Length Std
2. Bwd Packet Length Mean
3. Bwd Packet Length Max
4. Packet Length Std
5. Fwd IAT Std
6. Max Packet Length
7. Packet Length Variance
8. Idle Min
9. Idle Mean
10. Average Packet Size

---

# Current Status

Completed:
- [x] Raw CICIDS2017 loading
- [x] Dataset quality analysis
- [x] NaN/Infinity analysis
- [x] Duplicate analysis
- [x] Conflicting feature-vector analysis
- [x] Global conflict removal
- [x] Feature anomaly analysis
- [x] Constant feature removal
- [x] Exact duplicate feature removal
- [x] Train/test split
- [x] ANOVA feature ranking
- [x] Top-10 feature dataset
- [x] Top-20 feature dataset
- [x] Top-30 feature dataset
- [x] Top-40 feature dataset
- [x] Top-61 feature dataset
- [ ] Feature scaling
- [ ] CNN-GRU model
- [ ] Model evaluation
- [ ] Explainability analysis
- [ ] Zero-day/generalization experiments

---

# Repository Structure

```text
PS26-PAPER1/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_cleaning.py
│   ├── split_dataset.py
│   ├── anova_selection.py
│   ├── create_anova_datasets.py
│   └── scale_dataset.py
│
├── results/
├── notebooks/
├── docs/
├── models/
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Important

The raw CICIDS2017 dataset and generated CSV files are not stored directly in this repository because of their large size.

Each researcher should obtain the dataset separately and place the raw CSV files inside:
`data/raw/`

The preprocessing scripts can then regenerate the processed datasets.

### Research Direction

The current research investigates a lightweight and explainable deep-learning based NIDS.

The planned model architecture is based on CNN-GRU, with feature reduction through statistical selection and explainability through SHAP.

Further experiments will investigate model performance and generalization, including zero-day/leave-one-attack-out evaluation.
