import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/all_features_clean.csv")


# Load dataset

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# Prepare features and labels

target_col = "context"
subject_col = "person_id"

drop_cols = ["context", "person_id", "source_file", "start_index"]
X = df.drop(columns=drop_cols, errors="ignore")
y = df[target_col]

# Encode context labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

subjects = df[subject_col].unique()

print("\nSubjects:", subjects)

results = []


# Leave-One-Subject-Out loop

for test_subject in subjects:
    train_mask = df[subject_col] != test_subject
    test_mask = df[subject_col] == test_subject

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append({
        "subject": test_subject,
        "accuracy": acc,
        "num_test_windows": len(y_test)
    })

    print("\n==============================")
    print("Test subject:", test_subject)
    print("Number of test windows:", len(y_test))
    print("Accuracy:", round(acc, 3))

    # Only report classes that actually appear in this subject
    unique_labels = sorted(set(y_test))
    label_names = [le.classes_[i] for i in unique_labels]

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=unique_labels,
            target_names=label_names,
            zero_division=0
        )
    )


# Final summary

results_df = pd.DataFrame(results)

print("\n==============================")
print("FINAL LOSO SUMMARY")
print("==============================")
print(results_df.to_string(index=False))

print("\nAverage LOSO Accuracy:", round(results_df["accuracy"].mean(), 3))