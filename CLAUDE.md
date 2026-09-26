# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Intelligent Network Intrusion Detection System (NIDS) research project (final-year paper, "PS26"). It builds a CNN-GRU deep learning classifier on the **CICIDS2017** dataset for 7-class network traffic classification (BENIGN, DoS Hulk, DDoS, PortScan, DoS GoldenEye, FTP-Patator, SSH-Patator), and investigates two research questions beyond raw accuracy:

1. **Zero-day generalization** via Leave-One-Attack-Out (LOAO) evaluation — train with one attack class fully excluded, test on it as "unseen."
2. **Data leakage / protocol validity** of CICIDS2017 — auditing how much train/test overlap in standard stratified splits inflates reported metrics.

`PROJECT_CONTEXT.md` is the authoritative, detailed source of truth for the pipeline, dataset stats, model architecture, and results — read it before making non-trivial changes. `README.md` has an older/shorter version of the same narrative.

## Environment & running scripts

- No package manager config beyond `requirements.txt` (pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, jupyter, joblib) — no build/lint/test tooling exists in this repo (no pytest, no linter config). **`tensorflow` and `keras` are required by the training/demo scripts but are missing from `requirements.txt`** — install separately if setting up a fresh environment.
- Scripts have no CLI arguments/argparse and no `if __name__ == "__main__"` guards in most cases — configuration (feature counts, paths, hyperparameters) is hardcoded at the top of each file as module-level constants. To change behavior, edit the constants directly, then run e.g.:
  ```bash
  python src/preprocessing.py
  ```
- **Path inconsistency to watch for**: most scripts derive the project root dynamically via `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, but several older scripts hardcode a Windows-only absolute path instead:
  `src/analyze_loao.py`, `src/inspect_overlap_mismatch.py`, `src/protocol_sanity_check.py`, `src/tsne_analysis.py`, `src/deep_leakage_check.py`, `src/final_results_table.py` all have `BASE_DIR = r"D:\PS26-PAPER1"` (or similar) baked in. These will fail or write to the wrong place on Linux/macOS or a differently-located checkout — fix the constant before running them in this environment.
- Large data artifacts are gitignored and expected to be regenerated locally, not pulled from git: `data/raw/*.csv`, `data/processed/*.csv` and nested `**/*.csv`, and all model weight formats (`models/*.keras`, `*.h5`, `*.pt`, `*.pth`, `*.joblib`, `*.pkl`). Only small tracked artifacts (feature rankings, class weights CSV, results reports/figures, notebooks) live in git. `data/raw/` must be populated manually from an external CICIDS2017 download — it is not fetched by any script here.
- Do not delete or regenerate existing files in `models/` or the processed `.npy` sequence tensors casually — they represent specific verified experiment runs referenced in the results reports; regenerating is expensive (multi-million-row CSVs, full retraining).

## Pipeline architecture

The scripts in `src/` form a strict linear pipeline; each stage reads the previous stage's output from `data/processed/`. Understanding this ordering matters because scripts are not idempotent-safe against being run out of order:

```
preprocessing.py            raw CICIDS2017 CSVs → cleaned master CSV (NaN/Inf removal, label
                             conflict resolution — same feature vector w/ different labels dropped)
        ↓
feature_cleaning.py         drops 8 constant + 9 exact-duplicate columns (78 → 61 usable features)
        ↓
split_dataset.py            stratified 80/20 train/test split (class proportions preserved)
        ↓
anova_selection.py          ANOVA F-test ranks all 61 features on the training set only
create_anova_datasets.py    materializes Top-10/20/30/40/61 feature-subset CSVs
        ↓
scale_dataset.py            StandardScaler fit on train only, applied to train+test (leakage-safe)
class_weights.py            inverse-frequency class weights for imbalanced training
        ↓
create_sequences.py /       2D rows → 3D sliding-window tensors (window=10, stride=10,
create_class_sequences.py   shape (N, 10, num_features)); "class" variant keeps windows class-pure
        ↓
train_cnn_gru.py            trains CNN(Conv1D×2 + BatchNorm + MaxPool)-GRU-Dense classifier
                             per feature subset → models/cnn_gru_topN_final.keras + report
        ↓
loao_experiment.py /        retrains with one attack class fully held out, evaluates it as
strict_loao.py              zero-day; strict_loao.py is the leakage-hardened variant
analyze_loao.py             aggregates per-class LOAO recall/F1 into summary plots/tables
        ↓
protocol_sanity_check.py    structural sanity checks across the pipeline's directories
deep_leakage_check.py       quantifies train/test feature-vector overlap (measured: ~21-26%
                             of rows share identical feature vectors across splits — this is
                             the project's key critique of CICIDS2017 evaluation protocol)
inspect_overlap_mismatch.py drills into specific label-mismatched overlapping flows
        ↓
final_results_table.py,     builds paper-ready comparison tables/figures from results/ CSVs
create_paper1_figures.py
        ↓
presentation_demo.py /      interactive CLI demos loading a pretrained .keras model + a
demo_intrusion_detection.py fixed .npy test set, classifying pre-selected sample indices
find_demo_attacks.py        (utility to locate demo-worthy sample indices in the test set)
```

Key design points to preserve when modifying this pipeline:
- **Leakage-safety discipline**: scaling and feature selection are always fit on the training split only, then applied to test — this is intentional and load-bearing for the paper's methodology; don't refit on combined data.
- **Two evaluation regimes report very differently on purpose**: the conventional stratified 80/20 test set shows near-100% accuracy, while LOAO (true zero-day) shows ~6% mean recall on unseen attacks. This gap is the project's central finding, not a bug — don't "fix" LOAO to match conventional numbers.
- Model naming convention: `models/cnn_gru_top{10,20,30,40}_final.keras`, LOAO checkpoints under `models/loao/`. The Top-40 model is the one used in the presentation demo.
- Class label integers are fixed throughout the codebase: `0=BENIGN, 1=DoS Hulk, 2=DDoS, 3=PortScan, 4=DoS GoldenEye, 5=FTP-Patator, 6=SSH-Patator`.
