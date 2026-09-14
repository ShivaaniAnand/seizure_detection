import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

EEG_DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/chbmit_features.csv")


# Load EEG dataset

print("Loading EEG seizure dataset...")
df = pd.read_csv(EEG_DATA_PATH)

X = df.drop(columns=["label", "subject", "edf_file", "window_start_sec", "window_end_sec"])
y = df["label"]

print("Dataset shape:", X.shape)
print("Label counts:")
print(y.value_counts())


# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train EEG seizure model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining EEG seizure model...")
model.fit(X_train, y_train)

# EEG-only predictions
y_pred_eeg = model.predict(X_test)


# Simulated context labels

rng = np.random.default_rng(42)

sim_context = []
for true_label in y_test:
    if true_label == "seizure":
        # assume seizures are more likely during Rest/Sleep
        ctx = rng.choice(["R", "S"], p=[0.5, 0.5])
    else:
        # non-seizure may occur during any activity
        ctx = rng.choice(["R", "S", "W", "M"], p=[0.3, 0.2, 0.25, 0.25])
    sim_context.append(ctx)

sim_context = np.array(sim_context)


# Context-aware post-processing

y_pred_context_aware = y_pred_eeg.copy()

for i in range(len(y_pred_context_aware)):
    if y_pred_context_aware[i] == "seizure" and sim_context[i] in ["W", "M"]:
        y_pred_context_aware[i] = "non_seizure"


# EEG-only results

print("\n==============================")
print("EEG-ONLY RESULTS")
print("==============================")
print("Accuracy:", round(accuracy_score(y_test, y_pred_eeg), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_eeg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_eeg, labels=["non_seizure", "seizure"]))


# Context-aware results

print("\n==============================")
print("CONTEXT-AWARE RESULTS")
print("==============================")
print("Accuracy:", round(accuracy_score(y_test, y_pred_context_aware), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_context_aware))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_context_aware, labels=["non_seizure", "seizure"]))


# False alarm comparison

cm_eeg = confusion_matrix(y_test, y_pred_eeg, labels=["non_seizure", "seizure"])
cm_ctx = confusion_matrix(y_test, y_pred_context_aware, labels=["non_seizure", "seizure"])

fp_eeg = cm_eeg[0, 1]
fp_ctx = cm_ctx[0, 1]

print("\n==============================")
print("FALSE ALARM COMPARISON")
print("==============================")
print("EEG-only false positives:", fp_eeg)
print("Context-aware false positives:", fp_ctx)
print("Reduction in false positives:", fp_eeg - fp_ctx)