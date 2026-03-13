import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/chbmit_features.csv")
MODEL_PATH = os.path.expanduser("~/Desktop/seizure_project/models/seizure_model.pkl")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# features
X = df.drop(columns=["label","subject","edf_file","window_start_sec","window_end_sec"])

# labels
y = df["label"]

print("Dataset shape:", X.shape)
print("Label counts:")
print(y.value_counts())

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))

# save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nSaved seizure model to:")
print(MODEL_PATH)