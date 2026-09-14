import pandas as pd
import numpy as np
import os
import glob

DATA_DIR = os.path.expanduser("~/Desktop/seizure_project/data")
OUTPUT_FILE = os.path.expanduser("~/Desktop/seizure_project/all_features_clean.csv")

WINDOW_SIZE = 100

def extract_features(df):

    features = {}

    # IR features
    features["ir_mean"] = df["ir"].mean()
    features["ir_std"] = df["ir"].std()
    features["ir_min"] = df["ir"].min()
    features["ir_max"] = df["ir"].max()
    features["ir_ptp"] = df["ir"].max() - df["ir"].min()

    # Accelerometer mean
    features["ax_mean"] = df["ax"].mean()
    features["ay_mean"] = df["ay"].mean()
    features["az_mean"] = df["az"].mean()

    # Accelerometer std
    features["ax_std"] = df["ax"].std()
    features["ay_std"] = df["ay"].std()
    features["az_std"] = df["az"].std()

    acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_sma"] = np.sum(acc_mag)

    # Gyroscope mean
    features["gx_mean"] = df["gx"].mean()
    features["gy_mean"] = df["gy"].mean()
    features["gz_mean"] = df["gz"].mean()

    # Gyroscope std
    features["gx_std"] = df["gx"].std()
    features["gy_std"] = df["gy"].std()
    features["gz_std"] = df["gz"].std()

    features["gyro_energy"] = np.sum(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

    return features


def process_file(file_path):

    df = pd.read_csv(file_path)

    filename = os.path.basename(file_path)

    person_id = filename.split("_")[0]
    context = filename.split("_")[1]

    windows = []

    for start in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):

        window = df.iloc[start:start+WINDOW_SIZE]

        feats = extract_features(window)

        feats["person_id"] = person_id
        feats["context"] = context

        windows.append(feats)

    return windows


def main():

    files = glob.glob(DATA_DIR + "/P*/P*.csv")

    print("Found", len(files), "files")

    all_rows = []

    for f in files:

        print("Processing:", f)

        rows = process_file(f)

        all_rows.extend(rows)

    dataset = pd.DataFrame(all_rows)

    dataset = dataset.drop_duplicates()

    dataset.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal dataset saved to:")
    print(OUTPUT_FILE)

    print("\nDataset shape:", dataset.shape)

    print("\nContext distribution:")
    print(dataset["context"].value_counts())

    print("\nSubjects:")
    print(dataset["person_id"].value_counts())


if __name__ == "__main__":
    main()