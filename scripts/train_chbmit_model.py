import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.expanduser(
    "~/Desktop/seizure_detection"
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "chbmit_features.csv"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMARY_RESULTS_PATH = os.path.join(
    RESULT_DIR,
    "chbmit_seizure_patient_results.csv"
)

PRIMARY_PREDICTIONS_PATH = os.path.join(
    RESULT_DIR,
    "chbmit_seizure_patient_predictions.csv"
)

PRIMARY_SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "chbmit_seizure_patient_summary.csv"
)

NEGATIVE_RESULTS_PATH = os.path.join(
    RESULT_DIR,
    "chbmit_seizure_free_subject_results.csv"
)

NEGATIVE_PREDICTIONS_PATH = os.path.join(
    RESULT_DIR,
    "chbmit_seizure_free_subject_predictions.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "seizure_model.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("CHB-MIT PATIENT-INDEPENDENT SEIZURE CLASSIFICATION")
print("=" * 70)

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

required_columns = [
    "label",
    "subject",
    "edf_file",
    "window_start_sec",
    "window_end_sec"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

df["subject"] = df["subject"].astype(str)

subjects = sorted(
    df["subject"].unique()
)

print("\nSubjects found:")
print(subjects)

print("\nOverall class distribution:")
print(df["label"].value_counts())

print("\nClass distribution by subject:")
print(
    pd.crosstab(
        df["subject"],
        df["label"]
    )
)


# ============================================================
# PREPARE FEATURES
# ============================================================

metadata_columns = [
    "label",
    "subject",
    "edf_file",
    "window_start_sec",
    "window_end_sec"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

X = df[feature_columns].copy()
y = df["label"].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

if X.isnull().any().any():

    print(
        "\nMissing feature values found. "
        "Replacing them with feature medians."
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

else:

    print(
        "\nNo missing feature values detected."
    )

print(
    "\nNumber of ML features:",
    len(feature_columns)
)

print("\nFeatures used:")

for feature in feature_columns:
    print(" -", feature)


# ============================================================
# IDENTIFY SEIZURE-CONTAINING AND SEIZURE-FREE SUBJECTS
# ============================================================

seizure_subjects = []

seizure_free_subjects = []

for subject in subjects:

    subject_labels = df.loc[
        df["subject"] == subject,
        "label"
    ]

    if (
        subject_labels
        == "seizure"
    ).any():

        seizure_subjects.append(
            subject
        )

    else:

        seizure_free_subjects.append(
            subject
        )


print("\n" + "=" * 70)
print("SUBJECT GROUPS")
print("=" * 70)

print(
    "\nSubjects containing seizure windows:"
)

print(
    seizure_subjects
)

print(
    "\nSubjects containing only non-seizure windows:"
)

print(
    seizure_free_subjects
)


# ============================================================
# PRIMARY PATIENT-INDEPENDENT SEIZURE EVALUATION
#
# Only subjects containing annotated seizure windows are used
# as held-out seizure-detection test subjects.
#
# ALL other subjects remain available for training.
# ============================================================

fold_results = []

all_primary_predictions = []

print("\n" + "=" * 70)
print("PRIMARY SEIZURE-DETECTION EVALUATION")
print("=" * 70)

for fold_number, test_subject in enumerate(
    seizure_subjects,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"FOLD {fold_number}/"
        f"{len(seizure_subjects)} "
        f"- TEST PATIENT: "
        f"{test_subject}"
    )

    print("-" * 70)

    train_mask = (
        df["subject"]
        != test_subject
    )

    test_mask = (
        df["subject"]
        == test_subject
    )

    X_train = X.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    training_subjects = sorted(
        df.loc[
            train_mask,
            "subject"
        ].unique()
    )

    print(
        "\nTraining patients:"
    )

    print(
        training_subjects
    )

    print(
        "\nTest patient:",
        test_subject
    )

    print(
        "\nTraining windows:",
        len(X_train)
    )

    print(
        "Testing windows:",
        len(X_test)
    )

    print(
        "\nTraining class distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTesting class distribution:"
    )

    print(
        y_test.value_counts()
    )


    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred
        )
    )

    seizure_precision = (
        precision_score(
            y_test,
            y_pred,
            pos_label="seizure",
            zero_division=0
        )
    )

    seizure_recall = (
        recall_score(
            y_test,
            y_pred,
            pos_label="seizure",
            zero_division=0
        )
    )

    seizure_f1 = (
        f1_score(
            y_test,
            y_pred,
            pos_label="seizure",
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "non_seizure",
            "seizure"
        ]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[
                "non_seizure",
                "seizure"
            ],
            zero_division=0
        )
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy,
            4
        )
    )

    print(
        "Seizure Precision:",
        round(
            seizure_precision,
            4
        )
    )

    print(
        "Seizure Recall / Sensitivity:",
        round(
            seizure_recall,
            4
        )
    )

    print(
        "Seizure F1:",
        round(
            seizure_f1,
            4
        )
    )

    print(
        "Specificity:",
        round(
            specificity,
            4
        )
    )

    print(
        "\nTN:",
        tn
    )

    print(
        "FP:",
        fp
    )

    print(
        "FN:",
        fn
    )

    print(
        "TP:",
        tp
    )


    # --------------------------------------------------------
    # Store fold results
    # --------------------------------------------------------

    fold_results.append({

        "test_subject":
            test_subject,

        "train_windows":
            len(X_train),

        "test_windows":
            len(X_test),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "seizure_precision":
            seizure_precision,

        "seizure_recall_sensitivity":
            seizure_recall,

        "seizure_f1":
            seizure_f1,

        "specificity":
            specificity,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)
    })


    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    prediction_df = (
        df.loc[
            test_mask,
            [
                "subject",
                "edf_file",
                "window_start_sec",
                "window_end_sec",
                "label"
            ]
        ].copy()
    )

    prediction_df[
        "prediction"
    ] = y_pred

    prediction_df[
        "correct"
    ] = (
        prediction_df[
            "label"
        ]
        ==
        prediction_df[
            "prediction"
        ]
    )

    all_primary_predictions.append(
        prediction_df
    )


# ============================================================
# PRIMARY FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

print("\n" + "=" * 70)
print("PRIMARY PATIENT-INDEPENDENT RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# MEAN ± STANDARD DEVIATION ACROSS SEIZURE PATIENTS
# ============================================================

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "seizure_precision",
    "seizure_recall_sensitivity",
    "seizure_f1",
    "specificity"
]

summary_rows = []

for metric in metric_columns:

    summary_rows.append({

        "metric":
            metric,

        "mean":
            results_df[
                metric
            ].mean(),

        "std":
            results_df[
                metric
            ].std()
    })

summary_df = pd.DataFrame(
    summary_rows
)

print("\n" + "=" * 70)
print("MEAN PERFORMANCE ACROSS SEIZURE-CONTAINING PATIENTS")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['metric']}: "
        f"{row['mean']:.4f} "
        f"+/- "
        f"{row['std']:.4f}"
    )


# ============================================================
# POOLED PRIMARY RESULTS
# ============================================================

primary_predictions_df = pd.concat(
    all_primary_predictions,
    ignore_index=True
)

y_true_primary = (
    primary_predictions_df[
        "label"
    ]
)

y_pred_primary = (
    primary_predictions_df[
        "prediction"
    ]
)

primary_cm = confusion_matrix(
    y_true_primary,
    y_pred_primary,
    labels=[
        "non_seizure",
        "seizure"
    ]
)

primary_tn, primary_fp, primary_fn, primary_tp = (
    primary_cm.ravel()
)

primary_accuracy = accuracy_score(
    y_true_primary,
    y_pred_primary
)

primary_balanced_accuracy = (
    balanced_accuracy_score(
        y_true_primary,
        y_pred_primary
    )
)

primary_precision = precision_score(
    y_true_primary,
    y_pred_primary,
    pos_label="seizure",
    zero_division=0
)

primary_recall = recall_score(
    y_true_primary,
    y_pred_primary,
    pos_label="seizure",
    zero_division=0
)

primary_f1 = f1_score(
    y_true_primary,
    y_pred_primary,
    pos_label="seizure",
    zero_division=0
)

primary_specificity = (
    primary_tn
    /
    (
        primary_tn
        +
        primary_fp
    )
)

print("\n" + "=" * 70)
print("POOLED SEIZURE-PATIENT PERFORMANCE")
print("=" * 70)

print(
    "\nPooled Confusion Matrix:"
)

print(
    primary_cm
)

print(
    "\nPooled Classification Report:"
)

print(
    classification_report(
        y_true_primary,
        y_pred_primary,
        labels=[
            "non_seizure",
            "seizure"
        ],
        zero_division=0
    )
)

print(
    "Pooled Accuracy:",
    round(
        primary_accuracy,
        4
    )
)

print(
    "Pooled Balanced Accuracy:",
    round(
        primary_balanced_accuracy,
        4
    )
)

print(
    "Pooled Seizure Precision:",
    round(
        primary_precision,
        4
    )
)

print(
    "Pooled Seizure Recall / Sensitivity:",
    round(
        primary_recall,
        4
    )
)

print(
    "Pooled Seizure F1:",
    round(
        primary_f1,
        4
    )
)

print(
    "Pooled Specificity:",
    round(
        primary_specificity,
        4
    )
)

print(
    "\nPooled TN:",
    primary_tn
)

print(
    "Pooled FP:",
    primary_fp
)

print(
    "Pooled FN:",
    primary_fn
)

print(
    "Pooled TP:",
    primary_tp
)


# ============================================================
# SEIZURE-FREE SUBJECT FALSE-POSITIVE ANALYSIS
#
# These subjects are NOT used for seizure sensitivity,
# precision, or F1 calculations.
#
# They are analyzed separately to measure false positives.
# ============================================================

negative_results = []

negative_predictions = []

print("\n" + "=" * 70)
print("SEIZURE-FREE SUBJECT FALSE-POSITIVE ANALYSIS")
print("=" * 70)

for test_subject in seizure_free_subjects:

    print("\n" + "-" * 70)

    print(
        "Test seizure-free patient:",
        test_subject
    )

    print("-" * 70)

    train_mask = (
        df["subject"]
        != test_subject
    )

    test_mask = (
        df["subject"]
        == test_subject
    )

    X_train = X.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "non_seizure",
            "seizure"
        ]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    total_windows = (
        len(y_test)
    )

    false_positive_rate = (
        fp
        /
        (
            tn
            +
            fp
        )
        if (
            tn
            +
            fp
        ) > 0
        else np.nan
    )

    specificity = (
        tn
        /
        (
            tn
            +
            fp
        )
        if (
            tn
            +
            fp
        ) > 0
        else np.nan
    )

    print(
        "\nNon-seizure windows:",
        total_windows
    )

    print(
        "Correct non-seizure predictions:",
        tn
    )

    print(
        "False seizure predictions:",
        fp
    )

    print(
        "False-positive rate:",
        round(
            false_positive_rate,
            4
        )
    )

    print(
        "Specificity:",
        round(
            specificity,
            4
        )
    )

    negative_results.append({

        "test_subject":
            test_subject,

        "non_seizure_windows":
            total_windows,

        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_positive_rate":
            false_positive_rate,

        "specificity":
            specificity
    })

    prediction_df = (
        df.loc[
            test_mask,
            [
                "subject",
                "edf_file",
                "window_start_sec",
                "window_end_sec",
                "label"
            ]
        ].copy()
    )

    prediction_df[
        "prediction"
    ] = y_pred

    negative_predictions.append(
        prediction_df
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    PRIMARY_RESULTS_PATH,
    index=False
)

summary_df.to_csv(
    PRIMARY_SUMMARY_PATH,
    index=False
)

primary_predictions_df.to_csv(
    PRIMARY_PREDICTIONS_PATH,
    index=False
)

negative_results_df = pd.DataFrame(
    negative_results
)

negative_results_df.to_csv(
    NEGATIVE_RESULTS_PATH,
    index=False
)

if negative_predictions:

    negative_predictions_df = pd.concat(
        negative_predictions,
        ignore_index=True
    )

    negative_predictions_df.to_csv(
        NEGATIVE_PREDICTIONS_PATH,
        index=False
    )


print("\n" + "=" * 70)
print("RESULT FILES SAVED")
print("=" * 70)

print(
    "\nPrimary fold results:"
)

print(
    PRIMARY_RESULTS_PATH
)

print(
    "\nPrimary predictions:"
)

print(
    PRIMARY_PREDICTIONS_PATH
)

print(
    "\nPrimary summary:"
)

print(
    PRIMARY_SUMMARY_PATH
)

print(
    "\nSeizure-free subject results:"
)

print(
    NEGATIVE_RESULTS_PATH
)

print(
    "\nSeizure-free subject predictions:"
)

print(
    NEGATIVE_PREDICTIONS_PATH
)


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
#
# This model is for later prototype use only.
# Its training performance is NOT reported in the paper.
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON ALL AVAILABLE DATA")
print("=" * 70)

final_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X,
    y
)

joblib.dump(
    final_model,
    MODEL_PATH
)

print(
    "\nFinal model saved to:"
)

print(
    MODEL_PATH
)

print(
    "\nDone."
)