import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/all_features_clean.csv")
RESULT_DIR = os.path.expanduser("~/Desktop/seizure_project/results")

os.makedirs(RESULT_DIR, exist_ok=True)


# Load dataset

df = pd.read_csv(DATA_PATH)

target_col = "context"
subject_col = "person_id"

drop_cols = ["context", "person_id", "source_file", "start_index"]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

subjects = sorted(df[subject_col].unique())

print("Subjects:", subjects)

loso_results = []
pers_results = []

loso_true = []
loso_pred = []

pers_true = []
pers_pred = []

for subject in subjects:

    subject_mask = df[subject_col] == subject
    others_mask = df[subject_col] != subject

    X_subject = X[subject_mask]
    y_subject = y_encoded[subject_mask]

    X_others = X[others_mask]
    y_others = y_encoded[others_mask]

    if len(X_subject) < 5:
        continue

    # calibration/test split
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_subject,
        y_subject,
        test_size=0.8,
        random_state=42,
        stratify=y_subject
    )


    # LOSO model
   
    loso_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    loso_model.fit(X_others, y_others)
    loso_preds = loso_model.predict(X_test)
    loso_acc = accuracy_score(y_test, loso_preds)

    loso_results.append({
        "subject": subject,
        "loso_acc": loso_acc
    })

    loso_true.extend(y_test)
    loso_pred.extend(loso_preds)


    # Personalized model
    
    X_train_personalized = pd.concat(
        [X_others.reset_index(drop=True), X_calib.reset_index(drop=True)],
        ignore_index=True
    )

    y_train_personalized = pd.concat(
        [pd.Series(y_others).reset_index(drop=True), pd.Series(y_calib).reset_index(drop=True)],
        ignore_index=True
    )

    pers_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pers_model.fit(X_train_personalized, y_train_personalized)
    pers_preds = pers_model.predict(X_test)
    pers_acc = accuracy_score(y_test, pers_preds)

    pers_results.append({
        "subject": subject,
        "personalized_acc": pers_acc
    })

    pers_true.extend(y_test)
    pers_pred.extend(pers_preds)


loso_df = pd.DataFrame(loso_results)
pers_df = pd.DataFrame(pers_results)

results_df = pd.merge(loso_df, pers_df, on="subject")
results_df["improvement"] = results_df["personalized_acc"] - results_df["loso_acc"]

results_csv = os.path.join(RESULT_DIR, "accuracy_summary.csv")
results_df.to_csv(results_csv, index=False)

print("\nResults saved to:", results_csv)
print(results_df)


# Plot 1 — LOSO vs Personalized accuracy

x = np.arange(len(results_df["subject"]))
width = 0.35

plt.figure(figsize=(11, 5))
plt.bar(x - width / 2, results_df["loso_acc"], width, label="LOSO")
plt.bar(x + width / 2, results_df["personalized_acc"], width, label="Personalized")

plt.xticks(x, results_df["subject"])
plt.ylim(0, 1.05)
plt.xlabel("Subject")
plt.ylabel("Accuracy")
plt.title("LOSO vs Personalized Accuracy by Subject")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(os.path.join(RESULT_DIR, "accuracy_comparison.pdf"), format="pdf")
plt.close()


# Confusion matrices

cm_loso = confusion_matrix(loso_true, loso_pred, labels=np.arange(len(classes)))
cm_pers = confusion_matrix(pers_true, pers_pred, labels=np.arange(len(classes)))

# Normalize row-wise
cm_loso_norm = cm_loso.astype("float") / cm_loso.sum(axis=1, keepdims=True)
cm_pers_norm = cm_pers.astype("float") / cm_pers.sum(axis=1, keepdims=True)

# Handle divide-by-zero safely
cm_loso_norm = np.nan_to_num(cm_loso_norm)
cm_pers_norm = np.nan_to_num(cm_pers_norm)


def plot_confusion_heatmap(cm_norm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion")

    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()

plot_confusion_heatmap(
    cm_loso_norm,
    classes,
    "Confusion Matrix — LOSO Model",
    os.path.join(RESULT_DIR, "confusion_matrix_loso.pdf")
)

plot_confusion_heatmap(
    cm_pers_norm,
    classes,
    "Confusion Matrix — Personalized Model",
    os.path.join(RESULT_DIR, "confusion_matrix_personalized.pdf")
)

print("\nPlots saved in:", RESULT_DIR)
print("Generated files:")
print("- accuracy_summary.csv")
print("- accuracy_comparison.pdf")
print("- confusion_matrix_loso.pdf")
print("- confusion_matrix_personalized.pdf")