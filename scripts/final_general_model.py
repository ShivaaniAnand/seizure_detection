import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.expanduser(
    "~/Desktop/seizure_project/all_features_clean.csv"
)

MODEL_DIR = os.path.expanduser(
    "~/Desktop/seizure_project/models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


# Features

drop_cols = ["context","person_id","source_file","start_index"]

X = df.drop(columns=drop_cols, errors="ignore")
y = df["context"]

print("Dataset shape:", X.shape)


# Encode labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:", list(le.classes_))


# Train model

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

print("Training finished.")


# Save model

model_path = os.path.join(MODEL_DIR, "context_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(le, encoder_path)

print("\nSaved model to:")
print(model_path)