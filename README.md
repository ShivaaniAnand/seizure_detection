#Context-Aware Personalized Seizure Detection for Wearable IoT Devices

This project presents a context-aware and personalized seizure monitoring framework using wearable sensors and lightweight machine learning models. The system combines physiological signals, motion sensing, and activity context recognition to improve seizure detection reliability and reduce false alarms.

The proposed system is designed for edge deployment on wearable IoT devices, enabling continuous monitoring without relying heavily on cloud infrastructure.

Project Overview

Epilepsy is a neurological disorder characterized by unpredictable seizures. Continuous monitoring is essential for improving patient safety. Traditional EEG monitoring systems are typically hospital-based and not suitable for long-term everyday use.

This project explores how wearable sensors and machine learning can be used to detect seizure-related patterns while incorporating activity context and personalization.

The framework integrates:

wearable physiological sensing

motion-based activity recognition

EEG-based seizure detection

context-aware decision logic

personalized machine learning models

System Architecture

The system consists of two main components:

Wearable Device

Arduino Nano ESP32

MAX30102 PPG sensor (heart signal acquisition)

BMI160 IMU sensor (accelerometer + gyroscope)

These sensors collect physiological and motion data from the user.

Edge Processing Unit

A monitoring computer performs:

signal preprocessing

feature extraction

activity context recognition

EEG-based seizure detection

context-aware decision making

If a seizure event is detected, the system generates an alert message through the monitoring interface.

Key Features

Wearable sensor integration

Activity context recognition (Rest, Sleep, Walk, Motion)

Personalized machine learning models

Lightweight Random Forest classifiers

EEG seizure detection using the CHB-MIT dataset

Context-aware filtering to reduce false alarms

Edge-based processing for real-time monitoring

Datasets Used
Wearable Sensor Dataset

Data collected from 13 participants performing different activity contexts:

Resting

Sleeping

Walking

Motion

Sensor signals include:

PPG signal

accelerometer (x, y, z)

gyroscope (x, y, z)

EEG Dataset

Seizure detection experiments were performed using the CHB-MIT Scalp EEG Database.

Dataset source:
https://physionet.org/content/chbmit/

The EEG dataset was processed to extract seizure and non-seizure segments for model training.

Machine Learning Pipeline

The system follows this processing pipeline:

Data acquisition

Signal windowing

Feature extraction

Activity recognition

Seizure detection

Context-aware filtering

Alert generation

Statistical features extracted include:

mean

standard deviation

variance

signal magnitude area

accelerometer magnitude

gyroscope energy

Random Forest models were used due to their robustness and suitability for lightweight embedded systems.

Experimental Evaluation

Two experimental setups were used:

Leave-One-Subject-Out (LOSO)

Evaluates how well the model generalizes to unseen subjects.

Average accuracy:
85.3%

Personalized Calibration

A small amount of subject-specific data is used during training.

Average accuracy improved to:
92.2%

Seizure Detection

EEG-based seizure detection was evaluated with and without activity context.

Results showed that context-aware filtering reduces false seizure alarms while maintaining comparable detection accuracy.

Future Work

Possible improvements include:

integration of real-time EEG sensors

larger participant datasets

mobile applications for alert notifications

long-term real-world testing

improved deep learning-based seizure detection

Author

Shivaani Anand
MSc Information Technology – Data Science
Halmstad University
