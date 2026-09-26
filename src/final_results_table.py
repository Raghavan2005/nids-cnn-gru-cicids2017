from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
LOAO_FILE = RESULTS_DIR / "loao" / "loao_summary.csv"

OUTPUT_DIR = RESULTS_DIR / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PAPER 1 FINAL RESULTS TABLE")
print("=" * 70)

# ============================================================
# NORMAL CNN-GRU RESULTS
# ============================================================

normal_results = pd.DataFrame([
    {
        "Feature_Set": "Top-10",
        "Features": 10,
        "Accuracy": 0.9995,
        "Balanced_Accuracy": 0.9986,
        "Macro_F1": 0.9931,
        "Weighted_F1": 0.9995
    },
    {
        "Feature_Set": "Top-20",
        "Features": 20,
        "Accuracy": 1.0000,
        "Balanced_Accuracy": 0.9991,
        "Macro_F1": 0.9985,
        "Weighted_F1": 1.0000
    },
    {
        "Feature_Set": "Top-30",
        "Features": 30,
        "Accuracy": 1.0000,
        "Balanced_Accuracy": 0.9991,
        "Macro_F1": 0.9989,
        "Weighted_F1": 1.0000
    },
    {
        "Feature_Set": "Top-40",
        "Features": 40,
        "Accuracy": 1.0000,
        "Balanced_Accuracy": 1.0000,
        "Macro_F1": 1.0000,
        "Weighted_F1": 1.0000
    }
])

print("\nNORMAL TEST RESULTS")
print("-" * 70)

print(
    normal_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save
normal_file = OUTPUT_DIR / "cnn_gru_feature_comparison.csv"

normal_results.to_csv(
    normal_file,
    index=False
)

print("\nSaved:")
print(normal_file)


# ============================================================
# LOAO RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOAO RESULTS")
print("=" * 70)

loao = pd.read_csv(LOAO_FILE)

print(
    loao[
        [
            "unseen_attack",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "threshold"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

mean_recall = loao["recall"].mean()
mean_f1 = loao["f1"].mean()

print("\nMean LOAO Recall:", f"{mean_recall:.4f}")
print("Mean LOAO F1    :", f"{mean_f1:.4f}")


# Save LOAO
loao_file = OUTPUT_DIR / "loao_results.csv"

loao.to_csv(
    loao_file,
    index=False
)

print("\nSaved:")
print(loao_file)


# ============================================================
# PAPER SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "Analysis": "Top-10 CNN-GRU",
        "Main_Result": "Macro F1 = 0.9931",
        "Interpretation": "Strong classification performance"
    },
    {
        "Analysis": "Top-20 CNN-GRU",
        "Main_Result": "Macro F1 = 0.9985",
        "Interpretation": "Performance improves with more features"
    },
    {
        "Analysis": "Top-30 CNN-GRU",
        "Main_Result": "Macro F1 = 0.9989",
        "Interpretation": "Excellent conventional test performance"
    },
    {
        "Analysis": "Top-40 CNN-GRU",
        "Main_Result": "Macro F1 = 1.0000",
        "Interpretation": "Highest conventional test performance"
    },
    {
        "Analysis": "LOAO",
        "Main_Result": f"Mean Recall = {mean_recall:.4f}",
        "Interpretation": "Unseen attack generalization remains weak"
    },
    {
        "Analysis": "t-SNE",
        "Main_Result": "Top-30 feature-space visualization",
        "Interpretation": "Visual analysis of class separation"
    },
    {
        "Analysis": "Protocol Check",
        "Main_Result": "Train/test overlap detected",
        "Interpretation": "Results require cautious interpretation"
    }
])

summary_file = OUTPUT_DIR / "paper1_results_summary.csv"

summary.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 70)
print("PAPER 1 SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\nSaved:")
print(summary_file)

print("\n" + "=" * 70)
print("FINAL RESULTS TABLE CREATION COMPLETE")
print("=" * 70)