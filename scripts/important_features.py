import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = os.path.expanduser(
"~/Desktop/seizure_project/models/context_model.pkl"
)

DATA_PATH = os.path.expanduser(
"~/Desktop/seizure_project/all_features_clean.csv"
)

RESULT_DIR = os.path.expanduser(
"~/Desktop/seizure_project/results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# Load model

print("Loading model...")
model = joblib.load(MODEL_PATH)


# Load dataset

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

drop_cols = ["context","person_id","source_file","start_index"]

X = df.drop(columns=drop_cols, errors="ignore")

feature_names = X.columns


# Get feature importance

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 10 important features:\n")
print(importance_df.head(10))


# Save table

csv_path = os.path.join(
RESULT_DIR,
"feature_importance.csv"
)

importance_df.to_csv(csv_path, index=False)


# Plot feature importance

plt.figure(figsize=(10,6))

top_features = importance_df.head(15)

plt.barh(
top_features["feature"],
top_features["importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.title("Top 15 Feature Importances")

plt.tight_layout()

plot_path = os.path.join(
RESULT_DIR,
"feature_importance.png"
)

plt.savefig(plot_path, dpi=300)

print("\nSaved results to:")
print(csv_path)
print(plot_path)