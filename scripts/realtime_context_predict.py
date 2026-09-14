import os
import time
import serial
import joblib
import numpy as np
import pandas as pd
from collections import deque


PORT = "/dev/cu.usbmodemE4B063AE26802"
BAUD = 115200
WINDOW_SIZE = 100   # 5 seconds at ~20 Hz
MODEL_PATH = os.path.expanduser( "~/Desktop/seizure_project/models/context_model_P01.pkl")


# LOAD MODEL

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


# FEATURE EXTRACTION

def extract_features(df):
    features = {}

    # IR features
    features["ir_mean"] = df["ir"].mean()
    features["ir_std"] = df["ir"].std()
    features["ir_min"] = df["ir"].min()
    features["ir_max"] = df["ir"].max()
    features["ir_ptp"] = df["ir"].max() - df["ir"].min()

    # Accelerometer means
    features["ax_mean"] = df["ax"].mean()
    features["ay_mean"] = df["ay"].mean()
    features["az_mean"] = df["az"].mean()

    # Accelerometer std
    features["ax_std"] = df["ax"].std()
    features["ay_std"] = df["ay"].std()
    features["az_std"] = df["az"].std()

    # Acceleration magnitude
    acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_sma"] = np.sum(acc_mag)

    # Gyroscope means
    features["gx_mean"] = df["gx"].mean()
    features["gy_mean"] = df["gy"].mean()
    features["gz_mean"] = df["gz"].mean()

    # Gyroscope std
    features["gx_std"] = df["gx"].std()
    features["gy_std"] = df["gy"].std()
    features["gz_std"] = df["gz"].std()

    # Gyro energy
    features["gyro_energy"] = np.sum(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

    return features


# CONNECT TO SERIAL

print(f"Connecting to serial port {PORT} ...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Connected. Waiting for data...")

buffer = deque(maxlen=WINDOW_SIZE)


# MAIN LOOP

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if not line:
        continue

    # skip header
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

        # works whether model returns strings or numeric labels
        if isinstance(pred, str):
            pred_label = pred
        else:
            label_map = {
                0: "M",
                1: "R",
                2: "S",
                3: "W"
            }
            pred_label = label_map[pred]

        print(
            f"Context: {pred_label} "
        )