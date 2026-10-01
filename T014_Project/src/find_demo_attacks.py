import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

Y_PATH = BASE_DIR / "data" / "processed" / "sequences" / "test_top_30_y.npy"

CLASS_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
}

y = np.load(Y_PATH)

print("=" * 60)
print("AVAILABLE TEST ATTACK SAMPLES")
print("=" * 60)

for class_id, class_name in CLASS_NAMES.items():
    indices = np.where(y == class_id)[0]

    if len(indices) > 0:
        print(f"\n{class_name} (class {class_id})")
        print(f"  Number of samples: {len(indices)}")
        print(f"  Example indices: {indices[:10].tolist()}")

print("\n" + "=" * 60)
print("ATTACKS EXCLUDING DDoS AND PortScan")
print("=" * 60)

excluded = {0, 2, 3}

found = False

for class_id, class_name in CLASS_NAMES.items():
    if class_id in excluded:
        continue

    indices = np.where(y == class_id)[0]

    if len(indices) > 0:
        found = True
        print(f"\n>>> {class_name}")
        print(f"Class ID: {class_id}")
        print(f"Samples: {len(indices)}")
        print(f"Use index: {indices[0]}")

if not found:
    print("\nNo other attack class is present in test_top_30_y.npy.")

print("\nDone.")