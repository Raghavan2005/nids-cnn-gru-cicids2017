from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "scaled" / "train_top_30_scaled.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "scaled" / "test_top_30_scaled.csv"

OUT_DIR = BASE_DIR / "results" / "protocol_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("INSPECTING TRAIN / TEST LABEL MISMATCHES")
print("TOP-30 FEATURES")
print("=" * 70)

print("\nLoading data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

features = [c for c in train.columns if c != "Label"]

print("Train:", train.shape)
print("Test :", test.shape)

print("\nCreating row signatures...")

train["_hash"] = pd.util.hash_pandas_object(
    train[features],
    index=False
)

test["_hash"] = pd.util.hash_pandas_object(
    test[features],
    index=False
)

# ------------------------------------------------------------
# Find hashes existing in both datasets
# ------------------------------------------------------------

common = set(train["_hash"]).intersection(set(test["_hash"]))

print("Common feature patterns:", len(common))

train_common = train[train["_hash"].isin(common)]
test_common = test[test["_hash"].isin(common)]

# ------------------------------------------------------------
# Build train/test label combinations
# ------------------------------------------------------------

train_labels = (
    train_common
    .groupby("_hash")["Label"]
    .apply(lambda x: sorted(set(x)))
)

test_labels = (
    test_common
    .groupby("_hash")["Label"]
    .apply(lambda x: sorted(set(x)))
)

mismatches = []

for h in common:

    tr_labels = train_labels.get(h, [])
    te_labels = test_labels.get(h, [])

    if tr_labels != te_labels:

        mismatches.append({
            "hash": h,
            "train_labels": ",".join(map(str, tr_labels)),
            "test_labels": ",".join(map(str, te_labels))
        })

mismatch_df = pd.DataFrame(mismatches)

print("\n" + "=" * 70)
print("MISMATCH SUMMARY")
print("=" * 70)

print("Mismatched feature patterns:", len(mismatch_df))

# ------------------------------------------------------------
# Label combination counts
# ------------------------------------------------------------

if len(mismatch_df) > 0:

    print("\nTRAIN → TEST LABEL COMBINATIONS")
    print("-" * 70)

    combinations = (
        mismatch_df
        .groupby(
            ["train_labels", "test_labels"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(combinations.to_string(index=False))

    # Save combination summary
    combination_file = OUT_DIR / "top30_label_mismatch_combinations.csv"

    combinations.to_csv(
        combination_file,
        index=False
    )

    print("\nSaved:")
    print(combination_file)

# ------------------------------------------------------------
# Save mismatch hashes
# ------------------------------------------------------------

mismatch_file = OUT_DIR / "top30_label_mismatches.csv"

mismatch_df.to_csv(
    mismatch_file,
    index=False
)

print("\nSaved:")
print(mismatch_file)

# ------------------------------------------------------------
# Show examples with actual labels
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)

if len(mismatch_df) > 0:

    example_hashes = mismatch_df["hash"].head(10).tolist()

    for i, h in enumerate(example_hashes, 1):

        tr_rows = train_common[
            train_common["_hash"] == h
        ]

        te_rows = test_common[
            test_common["_hash"] == h
        ]

        print(f"\nExample {i}")
        print("-" * 40)

        print(
            "Training labels:",
            sorted(tr_rows["Label"].unique().tolist())
        )

        print(
            "Testing labels :",
            sorted(te_rows["Label"].unique().tolist())
        )

else:

    print("No mismatches found.")

# ------------------------------------------------------------
# Save readable report
# ------------------------------------------------------------

report_file = OUT_DIR / "top30_mismatch_inspection.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("TOP-30 TRAIN/TEST LABEL MISMATCH INSPECTION\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Common feature patterns: {len(common)}\n"
    )

    f.write(
        f"Mismatched feature patterns: "
        f"{len(mismatch_df)}\n\n"
    )

    if len(mismatch_df) > 0:

        f.write("LABEL COMBINATIONS\n")
        f.write("-" * 60 + "\n")

        f.write(
            combinations.to_string(index=False)
        )

        f.write("\n\nEXAMPLES\n")
        f.write("-" * 60 + "\n")

        for i, h in enumerate(example_hashes, 1):

            tr_rows = train_common[
                train_common["_hash"] == h
            ]

            te_rows = test_common[
                test_common["_hash"] == h
            ]

            f.write(f"\nExample {i}\n")

            f.write(
                "Training labels: "
                + str(sorted(
                    tr_rows["Label"].unique().tolist()
                ))
                + "\n"
            )

            f.write(
                "Testing labels: "
                + str(sorted(
                    te_rows["Label"].unique().tolist()
                ))
                + "\n"
            )

print("\nSaved:")
print(report_file)

print("\n" + "=" * 70)
print("MISMATCH INSPECTION COMPLETE")
print("=" * 70)