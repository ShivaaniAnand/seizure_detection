import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = os.path.expanduser("~/Desktop/seizure_project/chbmit_features.csv")
RESULT_DIR = os.path.expanduser("~/Desktop/seizure_project/results")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label","subject","edf_file","window_start_sec","window_end_sec"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds, labels=["non_seizure","seizure"])

fig, ax = plt.subplots()

im = ax.imshow(cm, cmap="Greys")

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["non_seizure","seizure"])
ax.set_yticklabels(["non_seizure","seizure"])

ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("EEG Seizure Detection Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

os.makedirs(RESULT_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULT_DIR,"seizure_confusion_matrix.pdf"))

print("Saved confusion matrix.")