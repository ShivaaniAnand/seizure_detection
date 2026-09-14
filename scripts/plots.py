import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_DIR = os.path.expanduser("~/Desktop/seizure_detection")

PREDICTIONS_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "loso_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "pictures",
    "confusion_matrix_loso.pdf"
)

# Subjects used in the primary multi-context evaluation
MULTI_CONTEXT_SUBJECTS = [
    "P01", "P02", "P03", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13"
]

# Final class order used in the LOSO experiment
CLASS_ORDER = ["M", "R", "S", "W"]

DISPLAY_LABELS = [
    "Motion",
    "Resting",
    "Sleeping",
    "Walking"
]

print("Loading predictions:")
print(PREDICTIONS_FILE)

df = pd.read_csv(PREDICTIONS_FILE)

print("\nColumns:")
print(df.columns.tolist())

# Keep only the 10 multi-context participants
df = df[df["subject"].isin(MULTI_CONTEXT_SUBJECTS)].copy()

print("\nNumber of pooled test windows:", len(df))

y_true = df["true_context"]
y_pred = df["prediction"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_ORDER
)

print("\nConfusion Matrix:")
print(cm)

# Safety check against the final LOSO result
expected_cm = [
    [103, 0,   0,   2],
    [1,   158, 24,  6],
    [0,   71,  138, 2],
    [4,   12,  0,   145]
]

if cm.tolist() == expected_cm:
    print("\nConfusion matrix matches the final LOSO experiment.")
else:
    print("\nWARNING: Confusion matrix does not match the expected final result.")
    print("Expected:")
    print(expected_cm)

fig, ax = plt.subplots(figsize=(5.5, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=DISPLAY_LABELS
)

disp.plot(
    ax=ax,
    cmap="Greys",
    colorbar=False,
    values_format="d"
)

ax.set_title("LOSO Activity Recognition")
ax.set_xlabel("Predicted Activity")
ax.set_ylabel("True Activity")

plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

plt.savefig(
    OUTPUT_FILE,
    format="pdf",
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(OUTPUT_FILE)