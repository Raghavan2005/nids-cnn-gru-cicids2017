# Intelligent Network Intrusion Detection System (CNN-GRU on CICIDS2017)

**Team T014** - Final Year Project

A deep-learning network intrusion detection system (NIDS) that classifies network flows into 7 classes
(BENIGN, DoS Hulk, DDoS, PortScan, DoS GoldenEye, FTP-Patator, SSH-Patator), plus two research questions
beyond raw accuracy:

1. **Zero-day generalisation** - Leave-One-Attack-Out (LOAO): train with one attack family removed, test it as unseen.
2. **Protocol validity** - how much do identical flows shared between train and test inflate CICIDS2017 results?

> All code lives in `T014_Project/`; final submission files go in `T014_Submission/`.
> `T014_Project/PROJECT_CONTEXT.md` holds the full technical detail (cleaning steps, architecture, every result).
> This README is the quick tour: what is here, how to run it, and how to check each module.

## Key results

| Evaluation | Result |
|---|---|
| Standard stratified 80/20 test (Top-40 features) | Accuracy 1.0000, Macro-F1 1.0000 |
| Top-10 features only | Accuracy 0.9995, Macro-F1 0.9931 |
| **Zero-day (LOAO), mean over 6 unseen attacks** | **Recall 6.23%, F1 11.50%** |
| **Train/test leakage (Top-30 features)** | **21.23% of train rows and 25.90% of test rows share an identical feature vector across the split; 81 such patterns carry conflicting labels** |

The gap is the finding: near-perfect accuracy on the usual split, but very low recall on attacks the model has never seen,
and part of the "perfect" score comes from duplicated flows. Do not "fix" LOAO to match the conventional numbers.

Per-attack LOAO recall: DoS Hulk 2.44%, DDoS 5.51%, PortScan 6.02%, DoS GoldenEye 13.17%, FTP-Patator 2.53%, SSH-Patator 7.69%.
Source files: `results/final/`, `results/loao/`, `results/protocol_check/`, `results/cnn_gru_top{10,20,30,40}_report.txt`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
cd T014_Project
pip install -r requirements.txt
pip install tensorflow keras          # training + model inference
pip install PySide6 segno             # live demo app (desktop UI + QR code)
```

The raw **CICIDS2017** CSVs are not in the repo (too large). Download them separately into `data/raw/`.
Large processed CSVs and model weights are git-ignored and regenerated locally.

## Modules and how to check each one

Run every script from inside `T014_Project/` (`cd T014_Project` first). Each stage reads the previous stage's output from `data/processed/`.
Settings (paths, feature counts, hyper-parameters) are constants at the top of each file.

| # | Module | Command | What to look for |
|---|---|---|---|
| 1 | Cleaning | `python src/preprocessing.py` | NaN/Inf removed; conflicting-label duplicates dropped |
| 2 | Feature cleaning | `python src/feature_cleaning.py` | 8 constant + 9 duplicate columns dropped: 78 -> 61 features |
| 3 | Split | `python src/split_dataset.py` | 2,244,572 train / 561,143 test, class proportions preserved |
| 4 | ANOVA selection | `python src/anova_selection.py`, `python src/create_anova_datasets.py` | Ranking in `data/processed/feature_selection/`; Top-10/20/30/40/61 sets |
| 5 | Scaling + class weights | `python src/scale_dataset.py`, `python src/class_weights.py` | StandardScaler fit on **train only** |
| 6 | Sequences | `python src/create_class_sequences.py` | Tensors of shape `(N, 10, features)` (window 10, stride 10) |
| 7 | Training | `python src/train_cnn_gru.py` | Model in `models/`, report in `results/cnn_gru_top*_report.txt` |
| 8 | LOAO (zero-day) | `python src/loao_experiment.py`, `python src/analyze_loao.py` | `results/loao/loao_summary.csv` (mean recall ~6%) |
| 9 | Leakage audit | `python src/deep_leakage_check.py`, `python src/protocol_sanity_check.py` | `results/protocol_check/` (21.23% / 25.90% overlap) |
| 10 | Figures / tables | `python src/final_results_table.py`, `python src/create_paper1_figures.py` | `results/final/` |
| 11 | Live demo | `python webapp/app.py` | see below |

Pipeline check without the real dataset: `python src/generate_synthetic_data.py` writes small CSVs with the CICIDS2017 column
schema so every stage can be exercised end to end. Numbers from synthetic data verify the **mechanics only**,
not the findings above.

## Model

`Input (10, F)` -> Conv1D(64, k=3) -> BatchNorm -> MaxPool(2) -> Conv1D(128, k=3) -> BatchNorm -> GRU(64) -> Dropout(0.3)
-> Dense(64) -> Dropout(0.3) -> Dense(7, softmax). Adam (lr 1e-3), sparse categorical cross-entropy, inverse-frequency class
weights, EarlyStopping / ReduceLROnPlateau / ModelCheckpoint. Naming: `models/cnn_gru_top{10,20,30,40}_final.keras`;
Top-40 is the demo model.

## Live demo: NIDS Command Center

```bash
python webapp/app.py
```

One desktop window (PySide6). A phone on the same Wi-Fi scans the QR code, opens a page, and sends traffic to the PC.
The CNN-GRU model classifies each flow live and the PC blocks repeat attackers (the phone then gets HTTP 403).

The dashboard shows: connection QR + counters, traffic-per-second chart, a live log of every request (filter, pause, CSV export,
also saved to `webapp/logs/traffic.log`), top source IPs, defence controls, the model's class probabilities, and the research
results (standard test, LOAO recall, leakage audit).

Be clear about what is real:
- The phone's HTTP requests are **real**. The model, however, classifies **CICIDS2017 flow-feature sequences sampled from the test set**
  for the traffic type chosen on the phone - a browser cannot produce those flow features.
- **Simulated network**: a switchable background generator sends mostly normal flows from random IPs in the reserved
  documentation ranges (`203.0.113.x`, `198.51.100.x`, `192.0.2.x`) with occasional short attack bursts. Every such line is tagged
  `SIM` in the Origin column.
- The status pill at the top shows **SYNTHETIC** when the model and test data in `models/` and
  `data/processed/sequences_class/` are the stand-ins from `generate_synthetic_data.py`. For the real demo put
  `cnn_gru_top40_final.keras` and `test_top_40_X.npy` / `test_top_40_y.npy` there; the pill turns green.
- Set `NIDS_PORT` to change the port (default 8000).

CLI alternative: `python src/presentation_demo.py` (interactive prompt, Top-40 model).

## Repository layout

```text
T014_Project/
  src/            pipeline, training, LOAO, leakage audit, figures, CLI demos
  webapp/         NIDS Command Center (app.py = UI, server.py = detection engine + phone endpoint)
  results/        reports, LOAO and leakage outputs, figures (tracked)
  notebooks/      EDA and feature analysis
  data/           raw/ and processed/ (large files git-ignored)
  models/         trained .keras weights (git-ignored)
  requirements.txt, PROJECT_CONTEXT.md
T014_Submission/  T014_ProjectReport.pdf, T014_Paper.pdf, T014_DemoVideo.mp4
README.md         this file
```

## Limitations

- Flow-feature based and offline: it does not capture or parse raw packets.
- 7 classes from a 2017 dataset; supervised softmax classification cannot recognise attack families it was not trained on
  (this is what LOAO measures).
- ANOVA ranks features one at a time and ignores interactions.
- Results depend on the CICIDS2017 split protocol, which this project shows to contain duplicated flows.
