import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.expanduser(
    "~/Desktop/seizure_detection"
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "all_features_clean.csv"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    RESULT_DIR,
    "loso_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "loso_summary.csv"
)

PREDICTIONS_PATH = os.path.join(
    RESULT_DIR,
    "loso_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("WEARABLE ACTIVITY RECOGNITION - LOSO EVALUATION")
print("=" * 70)

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(
    DATA_PATH
)

print(
    "\nDataset shape:",
    df.shape
)

print(
    "\nColumns:"
)

print(
    list(df.columns)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "person_id",
    "context"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN IDENTIFIERS
# ============================================================

df["person_id"] = (
    df["person_id"]
    .astype(str)
)

df["context"] = (
    df["context"]
    .astype(str)
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "person_id",
    "context",
    "source_file",
    "start_index"
]

feature_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]

print(
    "\nNumber of ML features:",
    len(feature_columns)
)

print(
    "\nFeatures used:"
)

for feature in feature_columns:

    print(
        " -",
        feature
    )


# ============================================================
# CLEAN FEATURE VALUES
# ============================================================

df[feature_columns] = (
    df[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

if (
    df[feature_columns]
    .isnull()
    .any()
    .any()
):

    print(
        "\nMissing values detected."
    )

    print(
        "Replacing missing values with column medians."
    )

    df[feature_columns] = (
        df[feature_columns]
        .fillna(
            df[feature_columns]
            .median(
                numeric_only=True
            )
        )
    )

else:

    print(
        "\nNo missing feature values detected."
    )


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["encoded_context"] = (
    label_encoder.fit_transform(
        df["context"]
    )
)

class_names = list(
    label_encoder.classes_
)

print(
    "\nActivity classes:"
)

print(
    class_names
)


# ============================================================
# SUBJECT INFORMATION
# ============================================================

subjects = sorted(
    df["person_id"]
    .unique()
)

print(
    "\nSubjects:"
)

print(
    subjects
)

print(
    "\nActivity distribution by subject:"
)

print(
    pd.crosstab(
        df["person_id"],
        df["context"]
    )
)


# ============================================================
# LOSO EVALUATION
#
# Each participant is completely excluded from training and
# used only for testing in their fold.
# ============================================================

results = []

all_predictions = []

print(
    "\n" + "=" * 70
)

print(
    "STARTING LEAVE-ONE-SUBJECT-OUT EVALUATION"
)

print(
    "=" * 70
)


for fold_number, test_subject in enumerate(
    subjects,
    start=1
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"FOLD {fold_number}/{len(subjects)} "
        f"- TEST SUBJECT: {test_subject}"
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # Train / test masks
    # --------------------------------------------------------

    train_mask = (
        df["person_id"]
        != test_subject
    )

    test_mask = (
        df["person_id"]
        == test_subject
    )


    train_df = (
        df.loc[
            train_mask
        ]
        .copy()
    )

    test_df = (
        df.loc[
            test_mask
        ]
        .copy()
    )


    available_contexts = sorted(
        test_df[
            "context"
        ].unique()
    )

    number_of_contexts = len(
        available_contexts
    )


    print(
        "\nTest subject contexts:"
    )

    print(
        available_contexts
    )

    print(
        "Number of contexts:",
        number_of_contexts
    )

    print(
        "\nTraining windows:",
        len(train_df)
    )

    print(
        "Testing windows:",
        len(test_df)
    )

    print(
        "\nTest distribution:"
    )

    print(
        test_df[
            "context"
        ].value_counts()
    )


    # --------------------------------------------------------
    # Features / labels
    # --------------------------------------------------------

    X_train = (
        train_df[
            feature_columns
        ]
    )

    y_train = (
        train_df[
            "encoded_context"
        ]
    )

    X_test = (
        test_df[
            feature_columns
        ]
    )

    y_test = (
        test_df[
            "encoded_context"
        ]
    )


    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
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


    # ========================================================
    # METRICS
    # ========================================================

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


    # --------------------------------------------------------
    # Macro F1 calculated over activity classes that actually
    # occur in this participant's test set.
    #
    # This avoids penalizing a participant for activities that
    # were never recorded for that participant.
    # --------------------------------------------------------

    labels_present = np.unique(
        y_test
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=labels_present,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nAccuracy:",
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
        "Macro F1:",
        round(
            macro_f1,
            4
        )
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_present,
            target_names=[
                label_encoder.inverse_transform(
                    [label]
                )[0]
                for label in labels_present
            ],
            zero_division=0
        )
    )


    # ========================================================
    # STORE FOLD RESULT
    # ========================================================

    results.append({

        "subject":
            test_subject,

        "number_of_contexts":
            number_of_contexts,

        "contexts":
            ",".join(
                available_contexts
            ),

        "train_windows":
            len(train_df),

        "test_windows":
            len(test_df),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1
    })


    # ========================================================
    # STORE HELD-OUT PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "subject":
            test_subject,

        "true_context":
            label_encoder.inverse_transform(
                y_test
            ),

        "prediction":
            label_encoder.inverse_transform(
                y_pred
            )
    })


    prediction_df[
        "correct"
    ] = (
        prediction_df[
            "true_context"
        ]
        ==
        prediction_df[
            "prediction"
        ]
    )


    all_predictions.append(
        prediction_df
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 70
)

print(
    "LOSO RESULTS - ALL SUBJECTS"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# ALL-SUBJECT SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MEAN RESULTS ACROSS ALL SUBJECTS"
)

print(
    "=" * 70
)


print(
    "\nMean Accuracy:"
)

print(
    f"{results_df['accuracy'].mean():.4f} "
    f"+/- "
    f"{results_df['accuracy'].std():.4f}"
)


print(
    "\nMean Balanced Accuracy:"
)

print(
    f"{results_df['balanced_accuracy'].mean():.4f} "
    f"+/- "
    f"{results_df['balanced_accuracy'].std():.4f}"
)


print(
    "\nMean Macro F1:"
)

print(
    f"{results_df['macro_f1'].mean():.4f} "
    f"+/- "
    f"{results_df['macro_f1'].std():.4f}"
)


# ============================================================
# MULTI-CONTEXT SUBJECTS
#
# Main result for the paper.
#
# Subjects with only one recorded activity are excluded from
# this aggregate because their classification task is much
# easier and can inflate overall activity-recognition accuracy.
# ============================================================

multi_context_df = (
    results_df[
        results_df[
            "number_of_contexts"
        ] >= 2
    ]
    .copy()
)


print(
    "\n" + "=" * 70
)

print(
    "MULTI-CONTEXT SUBJECT RESULTS"
)

print(
    "=" * 70
)


print(
    "\nSubjects included:"
)

print(
    multi_context_df[
        "subject"
    ].tolist()
)


print(
    "\nNumber of multi-context subjects:",
    len(
        multi_context_df
    )
)


if len(
    multi_context_df
) > 0:

    multi_accuracy_mean = (
        multi_context_df[
            "accuracy"
        ].mean()
    )

    multi_accuracy_std = (
        multi_context_df[
            "accuracy"
        ].std()
    )


    multi_balanced_mean = (
        multi_context_df[
            "balanced_accuracy"
        ].mean()
    )

    multi_balanced_std = (
        multi_context_df[
            "balanced_accuracy"
        ].std()
    )


    multi_f1_mean = (
        multi_context_df[
            "macro_f1"
        ].mean()
    )

    multi_f1_std = (
        multi_context_df[
            "macro_f1"
        ].std()
    )


    print(
        "\nLOSO Accuracy:"
    )

    print(
        f"{multi_accuracy_mean:.4f} "
        f"+/- "
        f"{multi_accuracy_std:.4f}"
    )


    print(
        "\nLOSO Balanced Accuracy:"
    )

    print(
        f"{multi_balanced_mean:.4f} "
        f"+/- "
        f"{multi_balanced_std:.4f}"
    )


    print(
        "\nLOSO Macro F1:"
    )

    print(
        f"{multi_f1_mean:.4f} "
        f"+/- "
        f"{multi_f1_std:.4f}"
    )


# ============================================================
# SINGLE-CONTEXT SUBJECTS
# ============================================================

single_context_df = (
    results_df[
        results_df[
            "number_of_contexts"
        ] == 1
    ]
    .copy()
)


print(
    "\n" + "=" * 70
)

print(
    "SINGLE-CONTEXT SUBJECTS"
)

print(
    "=" * 70
)


if len(
    single_context_df
) > 0:

    print(
        single_context_df[
            [
                "subject",
                "contexts",
                "accuracy",
                "balanced_accuracy",
                "macro_f1"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No single-context subjects."
    )


# ============================================================
# POOLED HELD-OUT PREDICTIONS
#
# Every prediction here comes from a fold where that subject
# was completely absent from training.
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


multi_context_subjects = (
    multi_context_df[
        "subject"
    ].tolist()
)


multi_predictions_df = (
    predictions_df[
        predictions_df[
            "subject"
        ].isin(
            multi_context_subjects
        )
    ]
    .copy()
)


pooled_accuracy = accuracy_score(
    multi_predictions_df[
        "true_context"
    ],
    multi_predictions_df[
        "prediction"
    ]
)


pooled_balanced_accuracy = (
    balanced_accuracy_score(
        multi_predictions_df[
            "true_context"
        ],
        multi_predictions_df[
            "prediction"
        ]
    )
)


pooled_macro_f1 = f1_score(
    multi_predictions_df[
        "true_context"
    ],
    multi_predictions_df[
        "prediction"
    ],
    labels=class_names,
    average="macro",
    zero_division=0
)


pooled_cm = confusion_matrix(
    multi_predictions_df[
        "true_context"
    ],
    multi_predictions_df[
        "prediction"
    ],
    labels=class_names
)


print(
    "\n" + "=" * 70
)

print(
    "POOLED MULTI-CONTEXT LOSO PERFORMANCE"
)

print(
    "=" * 70
)


print(
    "\nPooled Accuracy:",
    round(
        pooled_accuracy,
        4
    )
)


print(
    "Pooled Balanced Accuracy:",
    round(
        pooled_balanced_accuracy,
        4
    )
)


print(
    "Pooled Macro F1:",
    round(
        pooled_macro_f1,
        4
    )
)


print(
    "\nClass order:"
)

print(
    class_names
)


print(
    "\nPooled Confusion Matrix:"
)

print(
    pooled_cm
)


print(
    "\nPooled Classification Report:"
)

print(
    classification_report(
        multi_predictions_df[
            "true_context"
        ],
        multi_predictions_df[
            "prediction"
        ],
        labels=class_names,
        zero_division=0
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = [

    {
        "group":
            "all_subjects",

        "number_of_subjects":
            len(
                results_df
            ),

        "mean_accuracy":
            results_df[
                "accuracy"
            ].mean(),

        "std_accuracy":
            results_df[
                "accuracy"
            ].std(),

        "mean_balanced_accuracy":
            results_df[
                "balanced_accuracy"
            ].mean(),

        "std_balanced_accuracy":
            results_df[
                "balanced_accuracy"
            ].std(),

        "mean_macro_f1":
            results_df[
                "macro_f1"
            ].mean(),

        "std_macro_f1":
            results_df[
                "macro_f1"
            ].std()
    },

    {
        "group":
            "multi_context_subjects",

        "number_of_subjects":
            len(
                multi_context_df
            ),

        "mean_accuracy":
            multi_context_df[
                "accuracy"
            ].mean(),

        "std_accuracy":
            multi_context_df[
                "accuracy"
            ].std(),

        "mean_balanced_accuracy":
            multi_context_df[
                "balanced_accuracy"
            ].mean(),

        "std_balanced_accuracy":
            multi_context_df[
                "balanced_accuracy"
            ].std(),

        "mean_macro_f1":
            multi_context_df[
                "macro_f1"
            ].mean(),

        "std_macro_f1":
            multi_context_df[
                "macro_f1"
            ].std()
    },

    {
        "group":
            "pooled_multi_context_predictions",

        "number_of_subjects":
            len(
                multi_context_df
            ),

        "mean_accuracy":
            pooled_accuracy,

        "std_accuracy":
            np.nan,

        "mean_balanced_accuracy":
            pooled_balanced_accuracy,

        "std_balanced_accuracy":
            np.nan,

        "mean_macro_f1":
            pooled_macro_f1,

        "std_macro_f1":
            np.nan
    }
]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE FILES
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


print(
    "\n" + "=" * 70
)

print(
    "FILES SAVED"
)

print(
    "=" * 70
)


print(
    "\nLOSO per-subject results:"
)

print(
    RESULTS_PATH
)


print(
    "\nLOSO summary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nLOSO held-out predictions:"
)

print(
    PREDICTIONS_PATH
)


print(
    "\nDone."
)