# QAE-CyberDefense: Quantum Autoencoder for DDoS Detection

This repository contains a Hybrid Quantum-Classical Autoencoder implementation specifically designed to detect DDoS attacks using the UNSW-NB15 dataset. The system leverages quantum variational circuits to learn compressed representations of network traffic and identify cyber threats through anomaly detection.

# Prerequisites & Installation

Ensure you have Python 3.8+ installed. You can install the required libraries using the following command:
pip install requirements.txt
Note: If you encounter installation errors, please ensure your Python version is compatible with the library versions specified in the requirements file.

# Data Preprocessing (UNSW-NB15)
The model utilizes the UNSW-NB15 dataset, processed with the following steps:Feature Selection: Irrelevant features were removed to optimize quantum state encoding.

Scaling & Sampling: The dataset is scaled down to 10,000 samples for efficient quantum simulation.

Data Separation: Normal traffic and anomaly (DDoS) data are separated to facilitate unsupervised learning.

Normalization: All numerical features are normalized to fit within the $[0, \pi]$ range for quantum angle encoding.
