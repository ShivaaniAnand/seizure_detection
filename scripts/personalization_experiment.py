import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/all_features_clean.csv")


# Load dataset

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

target_col = "context"
subject_col = "person_id"

drop_cols = ["context", "person_id", "source_file", "start_index"]
X = df.drop(columns=drop_cols, errors="ignore")
y = df[target_col]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

subjects = df[subject_col].unique()

print("\nSubjects:", subjects)

results = []


# Personalization loop

for test_subject in subjects:
    subject_mask = df[subject_col] == test_subject
    others_mask = df[subject_col] != test_subject

    X_subject = X[subject_mask]
    y_subject = y_encoded[subject_mask]

    X_others = X[others_mask]
    y_others = y_encoded[others_mask]

    # If subject has too little data, skip safely
    if len(X_subject) < 5:
        print(f"\nSkipping {test_subject} (too few samples)")
        continue

    # Split subject data into calibration (20%) and final test (80%)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_subject,
        y_subject,
        test_size=0.8,
        random_state=42,
        stratify=y_subject
    )

    
    # General model
    
    general_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    general_model.fit(X_others, y_others)
    y_pred_general = general_model.predict(X_test)
    general_acc = accuracy_score(y_test, y_pred_general)

    
    # Personalized model
    
    X_train_personalized = pd.concat(
        [X_others.reset_index(drop=True), X_calib.reset_index(drop=True)],
        ignore_index=True
    )

    y_train_personalized = pd.concat(
        [pd.Series(y_others).reset_index(drop=True), pd.Series(y_calib).reset_index(drop=True)],
        ignore_index=True
    )

    personalized_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    personalized_model.fit(X_train_personalized, y_train_personalized)
    y_pred_personalized = personalized_model.predict(X_test)
    personalized_acc = accuracy_score(y_test, y_pred_personalized)

    improvement = personalized_acc - general_acc

    results.append({
        "subject": test_subject,
        "general_acc": general_acc,
        "personalized_acc": personalized_acc,
        "improvement": improvement,
        "test_windows": len(y_test),
        "calibration_windows": len(y_calib)
    })

    print("\n==============================")
    print("Test subject:", test_subject)
    print("General accuracy:", round(general_acc, 3))
    print("Personalized accuracy:", round(personalized_acc, 3))
    print("Improvement:", round(improvement, 3))
    print("Calibration windows:", len(y_calib))
    print("Test windows:", len(y_test))


# Final summary

results_df = pd.DataFrame(results)

print("\n==============================")
print("PERSONALIZATION SUMMARY")
print("==============================")
print(results_df.to_string(index=False))

print("\nAverage General Accuracy:", round(results_df["general_acc"].mean(), 3))
print("Average Personalized Accuracy:", round(results_df["personalized_acc"].mean(), 3))
print("Average Improvement:", round(results_df["improvement"].mean(), 3))