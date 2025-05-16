#!/usr/bin/env python3
"""
Model training script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Loads preprocessed data from a specific DVC version
2. Trains a Gradient Boosting model (XGBoost)
3. Performs hyperparameter tuning
4. Tracks experiments with MLflow
5. Registers the best model

Usage:
    python train.py --data-rev <DVC_REVISION>
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import mlflow
import mlflow.xgboost
import subprocess
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('model-training')

# Constants
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_FILE_TRAIN = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_DATA_FILE_VAL = PROCESSED_DATA_DIR / "val.csv"
MODELS_DIR = Path("models")
load_dotenv()

print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("MLFLOW_S3_ENDPOINT_URL:", os.getenv("MLFLOW_S3_ENDPOINT_URL"))

def parse_args():
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('--data-rev', type=str, required=False, default="HEAD",
                        help='(Optional) DVC revision/version of the processed data to use')
    return parser.parse_known_args()[0]


def setup_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("credit-card-fraud-detection")


def load_data(data_rev):
    logger.info(f"Pulling data from DVC revision: {data_rev}")
    subprocess.run(["dvc", "pull", "--force", PROCESSED_DATA_FILE_TRAIN.as_posix()], check=True)
    subprocess.run(["dvc", "pull", "--force", PROCESSED_DATA_FILE_VAL.as_posix()], check=True)
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]
    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0]
    }

    logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10,
                                scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logger.info(f"Best hyperparameters: {search.best_params_}")
    return best_model, search.best_params_


def evaluate_model(model, X_val, y_val):
    logger.info("Evaluating model on validation data...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "avg_precision": average_precision_score(y_val, y_proba)
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def log_to_mlflow(model, params, metrics):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "fraud-detection-model")


def save_model(model):
    logger.info("Saving model to disk...")
    joblib.dump(model, MODELS_DIR / "model.joblib")


def main():
    args = parse_args()
    logger.info(f"Starting model training pipeline with data revision: {args.data_rev}")

    setup_directories()
    setup_mlflow()
    X_train, y_train, X_val, y_val = load_data(args.data_rev)
    model, best_params = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_val, y_val)
    log_to_mlflow(model, best_params, metrics)
    save_model(model)

    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()