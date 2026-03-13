import os
import time
import serial
import joblib
import numpy as np
import pandas as pd
from collections import deque
from sklearn.ensemble import RandomForestClassifier

PORT = "/dev/cu.usbmodemE4B063AE26802"
BAUD = 115200

WINDOW_SIZE = 100

DATA_PATH = os.path.expanduser(
    "~/Desktop/seizure_project/all_features_clean.csv"
)

MODEL_DIR = os.path.expanduser(
    "~/Desktop/seizure_project/models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# FEATURE EXTRACTION

def extract_features(df):

    features = {}

    features["ir_mean"] = df["ir"].mean()
    features["ir_std"] = df["ir"].std()
    features["ir_min"] = df["ir"].min()
    features["ir_max"] = df["ir"].max()
    features["ir_ptp"] = df["ir"].max() - df["ir"].min()

    features["ax_mean"] = df["ax"].mean()
    features["ay_mean"] = df["ay"].mean()
    features["az_mean"] = df["az"].mean()

    features["ax_std"] = df["ax"].std()
    features["ay_std"] = df["ay"].std()
    features["az_std"] = df["az"].std()

    acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_sma"] = np.sum(acc_mag)

    features["gx_mean"] = df["gx"].mean()
    features["gy_mean"] = df["gy"].mean()
    features["gz_mean"] = df["gz"].mean()

    features["gx_std"] = df["gx"].std()
    features["gy_std"] = df["gy"].std()
    features["gz_std"] = df["gz"].std()

    features["gyro_energy"] = np.sum(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

    return features


# LOAD DATASET

df = pd.read_csv(DATA_PATH)

drop_cols = ["context","person_id","source_file","start_index"]

X_general = df.drop(columns=drop_cols, errors="ignore")
y_general = df["context"]


person_id = input("Enter user ID (example P14): ").strip()

print("\nStarting calibration for", person_id)
print("Stay still for ~30 seconds...\n")


# SERIAL CONNECTION

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

buffer = deque(maxlen=WINDOW_SIZE)

calibration_features = []

start_time = time.time()


# CALIBRATION PHASE

while time.time() - start_time < 30:

    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if not line:
        continue

    if line.startswith("time_ms"):
        continue

    values = line.split(",")

    if len(values) != 8:
        continue

    try:

        row = {
            "time_ms": float(values[0]),
            "ir": float(values[1]),
            "ax": float(values[2]),
            "ay": float(values[3]),
            "az": float(values[4]),
            "gx": float(values[5]),
            "gy": float(values[6]),
            "gz": float(values[7]),
        }

    except:
        continue

    buffer.append(row)

    if len(buffer) == WINDOW_SIZE:

        df_window = pd.DataFrame(buffer)

        feat = extract_features(df_window)

        calibration_features.append(feat)

print("\nCalibration finished.")



# TRAIN PERSONALIZED MODEL

X_person = pd.DataFrame(calibration_features)

y_person = ["R"] * len(X_person)  # assume resting calibration

X_train = pd.concat([X_general, X_person])
y_train = pd.concat([y_general, pd.Series(y_person)])

print("Training personalized model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

model_path = os.path.join(
    MODEL_DIR,
    f"context_model_{person_id}.pkl"
)

joblib.dump(model, model_path)

print("Saved personalized model:", model_path)


# REALTIME PREDICTION

print("\nStarting realtime prediction...\n")

while True:

    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if not line:
        continue

    if line.startswith("time_ms"):
        continue

    values = line.split(",")

    if len(values) != 8:
        continue

    try:

        row = {
            "time_ms": float(values[0]),
            "ir": float(values[1]),
            "ax": float(values[2]),
            "ay": float(values[3]),
            "az": float(values[4]),
            "gx": float(values[5]),
            "gy": float(values[6]),
            "gz": float(values[7]),
        }

    except:
        continue

    buffer.append(row)

    if len(buffer) == WINDOW_SIZE:

        df_window = pd.DataFrame(buffer)

        feat = extract_features(df_window)

        feat_df = pd.DataFrame([feat])

        pred = model.predict(feat_df)[0]

        print("Predicted context:", pred)