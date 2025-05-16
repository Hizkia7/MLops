#!/usr/bin/env python3
"""
Data preprocessing script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Loads a specific version of raw data from DVC
2. Splits data into train/validation/test sets
3. Normalizes features
4. Handles class imbalance
5. Saves processed datasets back to DVC
6. Logs preprocessing steps to MLflow

Usage:
    python preprocess.py --data-rev <DVC_REVISION>
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('data-preprocessing')

# Constants
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard-data.csv"

def parse_args():
    logger.info(f"Check parser")
    parser = argparse.ArgumentParser(description='Data preprocessing script')
    logger.info(f"Check parser 1")
    parser.add_argument('--data-rev', type=str, required=False, default="HEAD", help='(Optional) DVC revision/version of the raw data to use. Defaults to HEAD.')
    logger.info(f"Check parser 2")
    return parser.parse_known_args()[0]

def setup_directories():
    logger.info(f"Creating processed data directory: {PROCESSED_DATA_DIR}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Preprocessing")

def load_data(data_rev):
    logger.info(f"Checking out raw data at DVC revision: {data_rev}")
    # subprocess.run(["dvc", "checkout", RAW_DATA_FILE.as_posix(), "--rev", data_rev], check=True)
    subprocess.run(["git", "checkout", data_rev], check=True)
    subprocess.run(["dvc", "pull", RAW_DATA_FILE.as_posix()], check=True)
    logger.info(f"Loading dataset from {RAW_DATA_FILE}")
    return pd.read_csv(RAW_DATA_FILE)

def analyze_data(df):
    stats = {
        "num_rows": len(df),
        "num_features": df.shape[1],
        "num_fraud": df[df["Class"] == 1].shape[0],
        "num_normal": df[df["Class"] == 0].shape[0],
    }
    mlflow.log_metrics(stats)
    logger.info(f"Data summary: {stats}")
    return stats

def preprocess_data(df):
    logger.info("Splitting features and labels...")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    logger.info("Splitting into train/validation/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.22, random_state=42)

    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val =  scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info("Applying Downsampling to balance the dataset...")
    undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    train_df = pd.DataFrame(X_resampled)
    train_df["Class"] = y_resampled.values

    val_df = pd.DataFrame(X_val)
    val_df["Class"] = y_val.values

    test_df = pd.DataFrame(X_test)
    test_df["Class"] = y_test.values

    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df):
    logger.info("Saving processed datasets...")
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Tracking processed data with DVC...")
    # subprocess.run(["dvc", "commit", train_path.as_posix()], check=True)
    # subprocess.run(["dvc", "commit", str(val_path)], check=True)
    # subprocess.run(["dvc", "commit", str(test_path)], check=True)
    subprocess.run(["dvc", "commit", "preprocess", "--force"], check=True)
    time.sleep(10)
    subprocess.run(["git", "add", "."], check=True)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:  # there are staged changes
        subprocess.run(["git", "commit", "-m", "Add processed datasets"], check=True)
    else:
        print("No changes to commit.")
    subprocess.run(["dvc", "push"], check=True)

def log_to_mlflow(stats, train_df, val_df, test_df):
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("val_size", len(val_df))
    mlflow.log_param("test_size", len(test_df))
    mlflow.log_metrics({
        "class_ratio_train": train_df["Class"].mean(),
        "class_ratio_val": val_df["Class"].mean(),
        "class_ratio_test": test_df["Class"].mean()
    })

def main():
    args = parse_args()
    logger.info(f"Starting data preprocessing pipeline with data revision: {args.data_rev}")
    setup_directories()
    setup_mlflow()
    with mlflow.start_run():
        df = load_data(args.data_rev)
        stats = analyze_data(df)
        train_df, val_df, test_df = preprocess_data(df)
        save_processed_data(train_df, val_df, test_df)
        log_to_mlflow(stats, train_df, val_df, test_df)
    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()