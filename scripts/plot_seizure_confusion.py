import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_DIR = os.path.expanduser("~/Desktop/seizure_detection")

PREDICTIONS_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "chbmit_seizure_patient_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "pictures",
    "seizure_confusion_matrix.pdf"
)

print("Loading predictions:")
print(PREDICTIONS_FILE)

df = pd.read_csv(PREDICTIONS_FILE)

print("\nColumns:")
print(df.columns.tolist())

print("\nUnique label values:")
print(df["label"].unique())

print("\nUnique prediction values:")
print(df["prediction"].unique())


# ------------------------------------------------------------
# Convert labels into integers
# ------------------------------------------------------------

label_mapping = {
    "non_seizure": 0,
    "seizure": 1
}

df["label_clean"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .map(label_mapping)
)

df["prediction_clean"] = (
    df["prediction"]
    .astype(str)
    .str.strip()
    .map(label_mapping)
)

if df["label_clean"].isna().any():
    raise ValueError(
        f"Unexpected true-label values: {df['label'].unique()}"
    )

if df["prediction_clean"].isna().any():
    raise ValueError(
        f"Unexpected prediction values: {df['prediction'].unique()}"
    )

df["label_clean"] = df["label_clean"].astype(int)
df["prediction_clean"] = df["prediction_clean"].astype(int)

print("\nCleaned true labels:")
print(df["label_clean"].value_counts())

print("\nCleaned predictions:")
print(df["prediction_clean"].value_counts())


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

y_true = df["label_clean"]
y_pred = df["prediction_clean"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# Check against final experiment
# ------------------------------------------------------------

expected_cm = [
    [598, 55],
    [98, 197]
]

if cm.tolist() == expected_cm:
    print(
        "\nSUCCESS: Confusion matrix matches "
        "the final patient-independent EEG experiment."
    )
else:
    print("\nWARNING: Matrix does not match expected result.")

    print("\nExpected:")
    print(expected_cm)

    print("\nObtained:")
    print(cm.tolist())


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.2, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Seizure", "Seizure"]
)

disp.plot(
    ax=ax,
    cmap="Greys",
    colorbar=False,
    values_format="d"
)

ax.set_title(
    "Patient-Independent EEG Seizure Classification"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

plt.tight_layout()

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    format="pdf",
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(OUTPUT_FILE)