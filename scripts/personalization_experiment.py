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
    "personalization_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "personalization_summary.csv"
)

PREDICTIONS_PATH = os.path.join(
    RESULT_DIR,
    "personalization_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CALIBRATION_FRACTION = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("WEARABLE ACTIVITY PERSONALIZATION EXPERIMENT")
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
# CHECK REQUIRED COLUMNS
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
# CLEAN BASIC IDENTIFIERS
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
    "\nNumber of features:",
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
# CLEAN FEATURE DATA
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
        "\nMissing feature values detected."
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
# DATASET DISTRIBUTION
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
# TEMPORAL CALIBRATION / TEST SPLIT
#
# For each subject and each available context:
#
# First 20% of windows  -> calibration
# Remaining 80%         -> testing
#
# Windows are NOT randomly shuffled.
# ============================================================

def split_subject_temporally(
    subject_df,
    calibration_fraction=0.20
):

    calibration_parts = []

    test_parts = []

    contexts = sorted(
        subject_df[
            "context"
        ].unique()
    )

    for context in contexts:

        context_df = (
            subject_df[
                subject_df["context"]
                == context
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Preserve temporal ordering when metadata exists
        # ----------------------------------------------------

        if (
            "source_file"
            in context_df.columns
            and
            "start_index"
            in context_df.columns
        ):

            context_df = (
                context_df
                .sort_values(
                    [
                        "source_file",
                        "start_index"
                    ]
                )
            )

        elif (
            "start_index"
            in context_df.columns
        ):

            context_df = (
                context_df
                .sort_values(
                    "start_index"
                )
            )

        else:

            # Preserve original CSV order
            context_df = (
                context_df
                .sort_index()
            )


        number_of_windows = len(
            context_df
        )

        if number_of_windows < 2:

            print(
                f"WARNING: "
                f"Only {number_of_windows} window "
                f"available for context {context}."
            )

            continue


        calibration_count = max(
            1,
            int(
                np.floor(
                    number_of_windows
                    *
                    calibration_fraction
                )
            )
        )


        # Make sure at least one test window remains
        if calibration_count >= number_of_windows:

            calibration_count = (
                number_of_windows
                - 1
            )


        calibration_part = (
            context_df.iloc[
                :calibration_count
            ]
        )

        test_part = (
            context_df.iloc[
                calibration_count:
            ]
        )

        calibration_parts.append(
            calibration_part
        )

        test_parts.append(
            test_part
        )


    if (
        len(calibration_parts)
        == 0
        or
        len(test_parts)
        == 0
    ):

        return None, None


    calibration_df = pd.concat(
        calibration_parts
    )

    test_df = pd.concat(
        test_parts
    )

    return (
        calibration_df,
        test_df
    )


# ============================================================
# RUN PERSONALIZATION EXPERIMENT
# ============================================================

results = []

all_predictions = []

print("\n" + "=" * 70)
print("STARTING PERSONALIZATION EVALUATION")
print("=" * 70)


for subject_number, target_subject in enumerate(
    subjects,
    start=1
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"SUBJECT "
        f"{subject_number}/"
        f"{len(subjects)} "
        f"- {target_subject}"
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # Target subject data
    # --------------------------------------------------------

    subject_df = (
        df[
            df["person_id"]
            == target_subject
        ]
        .copy()
    )


    available_contexts = sorted(
        subject_df[
            "context"
        ].unique()
    )

    number_of_contexts = len(
        available_contexts
    )

    print(
        "\nAvailable contexts:"
    )

    print(
        available_contexts
    )

    print(
        "Number of contexts:",
        number_of_contexts
    )


    # --------------------------------------------------------
    # Temporal calibration/test split
    # --------------------------------------------------------

    calibration_df, test_df = (
        split_subject_temporally(
            subject_df,
            CALIBRATION_FRACTION
        )
    )


    if (
        calibration_df is None
        or
        test_df is None
    ):

        print(
            "Skipping subject because "
            "calibration/test split "
            "could not be created."
        )

        continue


    print(
        "\nCalibration windows:",
        len(calibration_df)
    )

    print(
        "Test windows:",
        len(test_df)
    )

    print(
        "\nCalibration distribution:"
    )

    print(
        calibration_df[
            "context"
        ].value_counts()
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
    # General training data
    #
    # All OTHER subjects only.
    # --------------------------------------------------------

    general_train_df = (
        df[
            df["person_id"]
            != target_subject
        ]
        .copy()
    )


    X_general_train = (
        general_train_df[
            feature_columns
        ]
    )

    y_general_train = (
        general_train_df[
            "encoded_context"
        ]
    )


    X_calibration = (
        calibration_df[
            feature_columns
        ]
    )

    y_calibration = (
        calibration_df[
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


    # ========================================================
    # GENERAL MODEL
    #
    # Train using all other subjects.
    # Target subject completely excluded.
    # ========================================================

    general_model = (
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    general_model.fit(
        X_general_train,
        y_general_train
    )

    general_predictions = (
        general_model.predict(
            X_test
        )
    )


    # ========================================================
    # PERSONALIZED MODEL
    #
    # Training =
    # all other subjects
    # +
    # first 20% temporal calibration windows
    # from target subject.
    # ========================================================

    X_personalized_train = pd.concat(
        [
            X_general_train,
            X_calibration
        ],
        axis=0
    )

    y_personalized_train = pd.concat(
        [
            y_general_train,
            y_calibration
        ],
        axis=0
    )


    personalized_model = (
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    personalized_model.fit(
        X_personalized_train,
        y_personalized_train
    )

    personalized_predictions = (
        personalized_model.predict(
            X_test
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    general_accuracy = (
        accuracy_score(
            y_test,
            general_predictions
        )
    )

    personalized_accuracy = (
        accuracy_score(
            y_test,
            personalized_predictions
        )
    )


    general_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            general_predictions
        )
    )

    personalized_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            personalized_predictions
        )
    )


    general_macro_f1 = (
        f1_score(
            y_test,
            general_predictions,
            average="macro",
            zero_division=0
        )
    )

    personalized_macro_f1 = (
        f1_score(
            y_test,
            personalized_predictions,
            average="macro",
            zero_division=0
        )
    )


    accuracy_improvement = (
        personalized_accuracy
        -
        general_accuracy
    )

    balanced_accuracy_improvement = (
        personalized_balanced_accuracy
        -
        general_balanced_accuracy
    )

    macro_f1_improvement = (
        personalized_macro_f1
        -
        general_macro_f1
    )


    # ========================================================
    # PRINT SUBJECT RESULTS
    # ========================================================

    print(
        "\nGeneral model:"
    )

    print(
        "Accuracy:",
        round(
            general_accuracy,
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            general_balanced_accuracy,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            general_macro_f1,
            4
        )
    )


    print(
        "\nPersonalized model:"
    )

    print(
        "Accuracy:",
        round(
            personalized_accuracy,
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            personalized_balanced_accuracy,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            personalized_macro_f1,
            4
        )
    )


    print(
        "\nImprovement:"
    )

    print(
        "Accuracy:",
        round(
            accuracy_improvement,
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_improvement,
            4
        )
    )

    print(
        "Macro F1:",
        round(
            macro_f1_improvement,
            4
        )
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nPersonalized Classification Report:"
    )

    print(
        classification_report(
            y_test,
            personalized_predictions,
            labels=np.arange(
                len(class_names)
            ),
            target_names=class_names,
            zero_division=0
        )
    )


    # ========================================================
    # STORE SUBJECT RESULTS
    # ========================================================

    results.append({

        "subject":
            target_subject,

        "number_of_contexts":
            number_of_contexts,

        "contexts":
            ",".join(
                available_contexts
            ),

        "calibration_windows":
            len(calibration_df),

        "test_windows":
            len(test_df),

        "general_accuracy":
            general_accuracy,

        "personalized_accuracy":
            personalized_accuracy,

        "accuracy_improvement":
            accuracy_improvement,

        "general_balanced_accuracy":
            general_balanced_accuracy,

        "personalized_balanced_accuracy":
            personalized_balanced_accuracy,

        "balanced_accuracy_improvement":
            balanced_accuracy_improvement,

        "general_macro_f1":
            general_macro_f1,

        "personalized_macro_f1":
            personalized_macro_f1,

        "macro_f1_improvement":
            macro_f1_improvement
    })


    # ========================================================
    # STORE WINDOW-LEVEL PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "subject":
            target_subject,

        "true_context":
            label_encoder.inverse_transform(
                y_test
            ),

        "general_prediction":
            label_encoder.inverse_transform(
                general_predictions
            ),

        "personalized_prediction":
            label_encoder.inverse_transform(
                personalized_predictions
            )
    })


    prediction_df[
        "general_correct"
    ] = (
        prediction_df[
            "true_context"
        ]
        ==
        prediction_df[
            "general_prediction"
        ]
    )


    prediction_df[
        "personalized_correct"
    ] = (
        prediction_df[
            "true_context"
        ]
        ==
        prediction_df[
            "personalized_prediction"
        ]
    )


    all_predictions.append(
        prediction_df
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 70
)

print(
    "PERSONALIZATION RESULTS"
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
# ALL SUBJECTS SUMMARY
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
    "General Accuracy:",
    round(
        results_df[
            "general_accuracy"
        ].mean(),
        4
    )
)

print(
    "Personalized Accuracy:",
    round(
        results_df[
            "personalized_accuracy"
        ].mean(),
        4
    )
)

print(
    "Mean Accuracy Improvement:",
    round(
        results_df[
            "accuracy_improvement"
        ].mean(),
        4
    )
)


print(
    "\nGeneral Balanced Accuracy:",
    round(
        results_df[
            "general_balanced_accuracy"
        ].mean(),
        4
    )
)

print(
    "Personalized Balanced Accuracy:",
    round(
        results_df[
            "personalized_balanced_accuracy"
        ].mean(),
        4
    )
)


print(
    "\nGeneral Macro F1:",
    round(
        results_df[
            "general_macro_f1"
        ].mean(),
        4
    )
)

print(
    "Personalized Macro F1:",
    round(
        results_df[
            "personalized_macro_f1"
        ].mean(),
        4
    )
)


# ============================================================
# MULTI-CONTEXT SUBJECT SUMMARY
#
# Primary summary for the paper.
#
# Subjects with only one activity are excluded from this
# aggregate because single-class testing can artificially
# produce very high accuracy.
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

    general_accuracy_mean = (
        multi_context_df[
            "general_accuracy"
        ].mean()
    )

    general_accuracy_std = (
        multi_context_df[
            "general_accuracy"
        ].std()
    )

    personalized_accuracy_mean = (
        multi_context_df[
            "personalized_accuracy"
        ].mean()
    )

    personalized_accuracy_std = (
        multi_context_df[
            "personalized_accuracy"
        ].std()
    )


    general_balanced_mean = (
        multi_context_df[
            "general_balanced_accuracy"
        ].mean()
    )

    personalized_balanced_mean = (
        multi_context_df[
            "personalized_balanced_accuracy"
        ].mean()
    )


    general_f1_mean = (
        multi_context_df[
            "general_macro_f1"
        ].mean()
    )

    personalized_f1_mean = (
        multi_context_df[
            "personalized_macro_f1"
        ].mean()
    )


    print(
        "\nGeneral Accuracy:"
    )

    print(
        f"{general_accuracy_mean:.4f} "
        f"+/- "
        f"{general_accuracy_std:.4f}"
    )


    print(
        "\nPersonalized Accuracy:"
    )

    print(
        f"{personalized_accuracy_mean:.4f} "
        f"+/- "
        f"{personalized_accuracy_std:.4f}"
    )


    print(
        "\nMean Accuracy Improvement:"
    )

    print(
        round(
            multi_context_df[
                "accuracy_improvement"
            ].mean(),
            4
        )
    )


    print(
        "\nGeneral Balanced Accuracy:"
    )

    print(
        round(
            general_balanced_mean,
            4
        )
    )


    print(
        "Personalized Balanced Accuracy:"
    )

    print(
        round(
            personalized_balanced_mean,
            4
        )
    )


    print(
        "\nGeneral Macro F1:"
    )

    print(
        round(
            general_f1_mean,
            4
        )
    )


    print(
        "Personalized Macro F1:"
    )

    print(
        round(
            personalized_f1_mean,
            4
        )
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
                "general_accuracy",
                "personalized_accuracy"
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
# SAVE SUMMARY
# ============================================================

summary_rows = [

    {
        "group":
            "all_subjects",

        "number_of_subjects":
            len(results_df),

        "general_accuracy":
            results_df[
                "general_accuracy"
            ].mean(),

        "personalized_accuracy":
            results_df[
                "personalized_accuracy"
            ].mean(),

        "accuracy_improvement":
            results_df[
                "accuracy_improvement"
            ].mean(),

        "general_balanced_accuracy":
            results_df[
                "general_balanced_accuracy"
            ].mean(),

        "personalized_balanced_accuracy":
            results_df[
                "personalized_balanced_accuracy"
            ].mean(),

        "general_macro_f1":
            results_df[
                "general_macro_f1"
            ].mean(),

        "personalized_macro_f1":
            results_df[
                "personalized_macro_f1"
            ].mean()
    },

    {
        "group":
            "multi_context_subjects",

        "number_of_subjects":
            len(
                multi_context_df
            ),

        "general_accuracy":
            (
                multi_context_df[
                    "general_accuracy"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "personalized_accuracy":
            (
                multi_context_df[
                    "personalized_accuracy"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "accuracy_improvement":
            (
                multi_context_df[
                    "accuracy_improvement"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "general_balanced_accuracy":
            (
                multi_context_df[
                    "general_balanced_accuracy"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "personalized_balanced_accuracy":
            (
                multi_context_df[
                    "personalized_balanced_accuracy"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "general_macro_f1":
            (
                multi_context_df[
                    "general_macro_f1"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            ),

        "personalized_macro_f1":
            (
                multi_context_df[
                    "personalized_macro_f1"
                ].mean()
                if len(
                    multi_context_df
                ) > 0
                else np.nan
            )
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


if len(
    all_predictions
) > 0:

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
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
    "\nPer-subject results:"
)

print(
    RESULTS_PATH
)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nPredictions:"
)

print(
    PREDICTIONS_PATH
)


print(
    "\nDone."
)