# Context-Aware Personalized Seizure Detection for Wearable IoT Devices

This project presents a context-aware and personalized seizure monitoring framework using wearable sensors and lightweight machine learning models. The system integrates physiological sensing, motion-based activity recognition, and EEG-based seizure detection to improve reliability and reduce false alarms in wearable health monitoring systems.

The proposed approach is designed for edge-based IoT environments, enabling continuous monitoring while minimizing reliance on cloud computing.

--------------------------------------------------

PROJECT OVERVIEW

Epilepsy is a neurological disorder characterized by recurrent and unpredictable seizures. Continuous monitoring is essential for improving patient safety and enabling timely medical intervention. However, traditional EEG monitoring systems are typically hospital-based and unsuitable for long-term everyday use.

This project explores how wearable IoT devices and machine learning models can support seizure monitoring in real-world environments. The system combines physiological signals, motion sensing, and contextual activity information to enhance seizure detection accuracy and reduce false alarms.

The proposed framework includes:

- Wearable physiological sensing
- Motion-based activity recognition
- EEG-based seizure detection
- Context-aware decision making
- Personalized machine learning models

--------------------------------------------------

SYSTEM ARCHITECTURE

The system consists of two main components.

Wearable Sensing Unit

The wearable prototype is built using:

- Arduino Nano ESP32
- MAX30102 PPG sensor for heart signal monitoring
- BMI160 IMU sensor for accelerometer and gyroscope measurements

These sensors continuously collect physiological and motion data from the user.

Edge Processing Unit

A monitoring computer processes the collected signals and performs:

- Signal preprocessing
- Feature extraction
- Activity context recognition
- Seizure detection using machine learning
- Context-aware decision logic

When a seizure event is detected, the system generates an alert notification for monitoring applications or caregivers.

--------------------------------------------------

DATASETS USED

Wearable Sensor Dataset

A wearable dataset was collected from 13 participants performing four activity contexts:

- Resting (R)
- Sleeping (S)
- Walking (W)
- Motion / active movement (M)

The recorded signals include:

- Infrared PPG signal
- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Activity context label

--------------------------------------------------

EEG DATASET

Seizure detection experiments were performed using the CHB-MIT Scalp EEG Database, a publicly available dataset commonly used in epilepsy research.

Dataset source:
https://physionet.org/content/chbmit/

The EEG recordings were processed to extract seizure and non-seizure segments for model training and evaluation.

--------------------------------------------------

MACHINE LEARNING PIPELINE

The system follows a multi-stage processing pipeline:

1. Data acquisition from wearable sensors
2. Signal segmentation into fixed-length windows
3. Statistical feature extraction
4. Activity context recognition
5. EEG-based seizure detection
6. Context-aware filtering
7. Alert generation

Extracted features include:

- Mean
- Standard deviation
- Minimum and maximum signal values
- Signal magnitude area
- Accelerometer magnitude
- Gyroscope energy

Random Forest classifiers were used because they are robust, computationally efficient, and suitable for resource-constrained embedded systems.

--------------------------------------------------

EXPERIMENTAL EVALUATION

Two experimental configurations were used to evaluate the system.

Leave-One-Subject-Out (LOSO)

LOSO validation evaluates the ability of the model to generalize to unseen subjects.

Average classification accuracy:
85.3%

Personalized Calibration

A small amount of subject-specific data is incorporated during training to adapt the model to individual physiological characteristics.

Average classification accuracy improved to:
92.2%

--------------------------------------------------

CONTEXT-AWARE SEIZURE DETECTION

EEG-based seizure detection was evaluated with and without contextual activity information. The results show that incorporating activity context reduces false seizure alarms while maintaining comparable detection accuracy.

This demonstrates the advantage of combining physiological monitoring with contextual activity information in wearable seizure detection systems.

--------------------------------------------------

REPOSITORY STRUCTURE

seizure-project
│
├── scripts
│   ├── feature_extraction.py
│   ├── train_context_model.py
│   ├── realtime_context_predict.py
│   ├── train_chbmit_model.py
│   └── combined_seizure_detection.py
│
├── datasets
│   ├── wearable_data
│   └── chbmit
│
├── models
│   ├── context_model.pkl
│   └── seizure_model.pkl
│
├── results
│   ├── confusion_matrix_loso.pdf
│   ├── confusion_matrix_personalized.pdf
│   └── accuracy_summary.csv
│
├── diagrams
│   ├── system_architecture
│   ├── activity_diagram
│   ├── sequence_diagram
│   └── deployment_diagram
│
└── README.md

--------------------------------------------------

TECHNOLOGIES USED

- Python
- Arduino / ESP32
- PlantUML
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- MNE
- WFDB

--------------------------------------------------

FUTURE WORK

Future improvements may include:

- Integration of real-time wearable EEG devices
- Larger and more diverse participant datasets
- Real-time mobile monitoring applications
- Long-term real-world testing
- Advanced deep learning-based seizure detection models

--------------------------------------------------

AUTHOR

Shivaani Anand  
MSc Information Technology – Data Science  
Halmstad University

--------------------------------------------------

LICENSE

This project is intended for research and educational purposes.
