import os
import csv
import time
import serial
from datetime import datetime

PORT = "/dev/cu.usbmodemE4B063AE26802"   # change if needed
BAUD = 115200

BASE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "seizure_project",
    "data"
)

def main():

    # Ask user info
    person_id = input("Enter Person ID (P01, P02...): ").strip().upper()
    context = input("Enter Context (R/W/M/S): ").strip().upper()

    if not person_id or not context:
        print("Person ID and Context required.")
        return

    # Create folder
    person_dir = os.path.join(BASE_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    # File name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{person_id}_{context}_{timestamp}.csv"
    filepath = os.path.join(person_dir, filename)

    print("\nSaving to:", filepath)
    print("Recording... Press CTRL+C to stop.\n")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header with label
        writer.writerow([
            "time_ms","ir","ax","ay","az","gx","gy","gz","label"
        ])

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if line:
                    values = line.split(",")
                    values.append(context)  # add label

                    print(values)
                    writer.writerow(values)

        except KeyboardInterrupt:
            print("\nLogging stopped.")

        finally:
            ser.close()


if __name__ == "__main__":
    main()