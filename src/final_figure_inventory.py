from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

FIGURE_DIRS = {
    "Final Results": PROJECT_DIR / "results" / "final",
    "Feature Analysis": PROJECT_DIR / "results" / "feature_analysis",
    "LOAO Analysis": PROJECT_DIR / "results" / "loao",
}

print("=" * 70)
print("PAPER 1 FINAL FIGURE INVENTORY")
print("=" * 70)

total = 0

for category, folder in FIGURE_DIRS.items():

    print("\n" + "-" * 70)
    print(category)
    print("-" * 70)

    if not folder.exists():
        print("[MISSING FOLDER]", folder)
        continue

    files = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]
    )

    if not files:
        print("No figure files found.")
        continue

    for i, file in enumerate(files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"{i}. {file.name} ({size_kb:.1f} KB)")
        total += 1

print("\n" + "=" * 70)
print(f"TOTAL FIGURES FOUND: {total}")
print("=" * 70)

print("\nRecommended Paper 1 figure structure:")
print("1. CNN-GRU System Architecture")
print("2. Feature-count performance comparison")
print("3. LOAO unseen-attack performance")
print("4. Top-30 t-SNE feature-space visualization")
print("5. Confusion matrix for selected/best conventional model")

print("\nNote:")
print("Do not create unnecessary figures just to increase the number of figures.")
print("Use figures that directly support the methodology and results.")