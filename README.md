# Transformer-Based Diffusion Anomaly Detection for Personal Finance

This repository contains an implementation of an advanced anomaly detection system designed to identify irregular spending patterns in personal finance transaction data. The system combines graph neural networks, transformer-based encoders, diffusion models, and Gaussian Mixture Models (GMM) to accurately detect anomalies in financial transactions.

## Features

* **Graph Convolutional Network (GCN):** Captures relational dependencies between transactions based on categories and payment methods.
* **Transformer Encoder:** Learns temporal and contextual patterns within transaction features.
* **Diffusion Model:** Performs noise-based reconstruction to model normal transaction behavior.
* **Gaussian Mixture Model (GMM):** Provides adaptive thresholding for anomaly score classification.
* **End-to-end PyTorch implementation** with data preprocessing and evaluation.

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* torch-geometric
* scikit-learn
* pandas
* numpy

Install dependencies with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn pandas numpy
```

### Dataset

The project uses a personal finance transaction dataset with the following features:

* Date, Category, Payment Method, DayOfWeek, Transaction Type, Income, Expense, Balance, etc.

Make sure to update the file path in the script before running.

### Running the Model

1. Preprocess the data with label encoding and normalization.
2. Build a transaction graph based on category and payment method similarities.
3. Train the model to learn normal transaction embeddings and reconstruct them with diffusion noise.
4. Use GMM to classify transactions as normal or anomalous.
5. Evaluate model performance using precision, recall, F1-score, and ROC-AUC metrics.

Run training and anomaly detection with:

```bash
python anomaly_detection.py
```

### Results

* The model outputs anomaly scores and flags top anomalies.
* Results are saved to `T-Diff_anomaly_results.csv`.
* Performance metrics are printed to the console.

## Project Structure

* `anomaly_detection.py` — main script with data processing, model definition, training, and evaluation.
* `T-Diff_anomaly_results.csv` — anomaly detection output file.
* `requirements.txt` — Python dependencies.


