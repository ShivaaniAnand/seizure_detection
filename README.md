# Context-Aware Personalized Seizure Detection for Wearable IoT Devices

This project presents a wearable seizure-monitoring framework that combines physiological sensing, motion-based activity recognition, and EEG-based seizure classification using lightweight machine learning models.

The current study evaluates wearable activity recognition and EEG seizure detection independently. The wearable recordings and CHB-MIT EEG recordings originate from different participants and are not synchronized. Integration of both components into a fully context-aware seizure-monitoring system is therefore considered future work.

---

## PROJECT OVERVIEW

Epilepsy is a neurological disorder characterized by recurrent and unpredictable seizures. Continuous monitoring can improve patient safety and support timely medical intervention. However, traditional EEG monitoring systems are generally hospital-based and are not suitable for continuous everyday monitoring.

This project investigates the use of wearable IoT sensors and lightweight machine learning models for seizure-monitoring applications.

The framework includes:

- Wearable physiological sensing
- Motion-based activity recognition
- EEG-based seizure classification
- Personalized activity-recognition models
- A proposed context-aware monitoring architecture

---

## SYSTEM ARCHITECTURE

The system contains two experimental components.

### Wearable Sensing Unit

The wearable prototype is built using:

- Arduino Nano ESP32
- MAX30102 PPG sensor
- BMI160 accelerometer and gyroscope sensor

The sensors continuously collect physiological and motion data.

The ESP32 performs sensor acquisition and transmits the measurements to a monitoring computer.

### Monitoring and Processing Unit

Python-based software on the monitoring computer performs:

- Data logging
- Signal segmentation
- Feature extraction
- Activity-context recognition
- Personalized calibration
- Machine-learning evaluation

EEG seizure classification is evaluated separately using the CHB-MIT dataset.

---

## WEARABLE DATASET

Wearable recordings were collected from 13 healthy participants.

The recorded activity contexts were:

- Resting (R)
- Sleeping (S)
- Walking (W)
- Motion / active movement (M)

Recorded signals include:

- Infrared PPG signal
- Accelerometer measurements (x, y, z)
- Gyroscope measurements (x, y, z)
- Activity-context label

The signals were divided into non-overlapping windows of 100 samples, corresponding to approximately 5 seconds at the acquisition rate used in the experiments.

A total of 21 wearable features were extracted for machine-learning evaluation.

---

## EEG DATASET

EEG seizure-classification experiments were performed using the CHB-MIT Scalp EEG Database.

Dataset source:

https://physionet.org/content/chbmit/

The raw CHB-MIT dataset is not included in this repository because of its large size.

Processed feature files and experimental results used in this project are included where applicable.

The EEG analysis uses statistical features extracted from seizure and non-seizure windows.

---

## MACHINE LEARNING PIPELINE

### Wearable Activity Recognition

The wearable processing pipeline consists of:

1. Sensor data acquisition
2. Data logging
3. Window segmentation
4. Statistical feature extraction
5. Activity-context classification
6. Subject-independent evaluation
7. Personalized calibration

A Random Forest classifier with 100 trees was used for activity recognition.

### EEG Seizure Classification

The EEG processing pipeline consists of:

1. CHB-MIT EEG preprocessing
2. Window segmentation
3. EEG feature extraction
4. Seizure/non-seizure classification
5. Patient-independent evaluation

The wearable activity and EEG seizure experiments are evaluated independently in the current study.

---

## EXPERIMENTAL EVALUATION

### Leave-One-Subject-Out Activity Recognition

Subject-independent activity recognition was evaluated using Leave-One-Subject-Out (LOSO) validation.

The primary analysis includes the 10 participants who recorded multiple activity contexts.

Mean subject-level results:

- Accuracy: **81.83% ± 13.08%**
- Balanced Accuracy: **82.74% ± 13.39%**
- Macro F1-score: **80.75% ± 14.76%**

The pooled LOSO evaluation across 666 held-out windows achieved:

- Accuracy: **81.68%**
- Balanced Accuracy: **84.29%**
- Macro F1-score: **83.99%**

---

## PERSONALIZED CALIBRATION

Personalization was evaluated by incorporating a small amount of subject-specific calibration data.

For each target participant:

- The first 20% of available samples for each context were used for calibration.
- The remaining 80% were used for evaluation.
- The general model was trained using data from the other participants.
- The personalized model additionally included the target participant's calibration data.

Across the multi-context participants:

- General-model accuracy: **83.03%**
- Personalized accuracy: **89.42%**
- Mean improvement: **6.39 percentage points**
- General balanced accuracy: **83.59%**
- Personalized balanced accuracy: **89.71%**
- General Macro F1-score: **81.57%**
- Personalized Macro F1-score: **85.63%**

The amount of improvement varied between participants.

---

## PATIENT-INDEPENDENT EEG SEIZURE CLASSIFICATION

Patient-independent seizure classification was evaluated using a subset of the CHB-MIT dataset.

Pooled held-out results:

- Accuracy: **83.86%**
- Balanced Accuracy: **79.18%**
- Seizure Precision: **78.17%**
- Seizure Sensitivity: **66.78%**
- Seizure F1-score: **72.03%**
- Specificity: **91.58%**

The wearable activity recordings and CHB-MIT EEG recordings are not synchronized. Therefore, the present experiments do not demonstrate that activity context reduces seizure false alarms.

Future synchronized multimodal recordings are required to evaluate this hypothesis.

---

## REPOSITORY STRUCTURE

```text
seizure_detection/
│
├── all_features_clean.csv
├── data_summary.csv
├── chbmit_features.csv
├── chbmit_seizure_events.csv
├── chbmit_window_index.csv
│
├── data/
│   ├── P01/
│   ├── P02/
│   ├── ...
│   └── P13/
│
├── diagrams/
│   ├── activity_diagram.png
│   ├── deployment_diagram.svg
│   ├── hardware_setup.pdf
│   ├── hardware_setup2.pdf
│   └── system_architecture.png
│
├── models/
│   ├── context_model.pkl
│   ├── context_model_P01.pkl
│   └── label_encoder.pkl
│
├── results/
│   ├── confusion_matrix_loso.pdf
│   ├── seizure_confusion_matrix.pdf
│   ├── loso_results.csv
│   ├── loso_summary.csv
│   ├── personalization_results.csv
│   ├── personalization_summary.csv
│   └── CHB-MIT evaluation results
│
├── scripts/
│   ├── logger.py
│   ├── data_pipeline.py
│   ├── loso_model.py
│   ├── personalization_experiment.py
│   ├── train_chbmit_model.py
│   ├── plot_seizure_confusion.py
│   └── other experimental scripts
│
└── README.md
