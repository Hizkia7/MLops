#!/usr/bin/env python3
"""
Model evaluation and validation script for Credit Card Fraud Detection MLOps Pipeline.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve
)

from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model-validation")

# Constants
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_FILE_TEST = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = Path("models")
VALIDATION_DIR = MODELS_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

PERFORMANCE_REQUIREMENTS = {
    "min_accuracy": 0.98,
    "min_precision": 0.85,
    "min_recall": 0.70,
    "min_f1_score": 0.75,
    "min_roc_auc": 0.95
}

class InferenceInput(BaseModel):
    inputs: Dict[str, float]

def parse_args():
    parser = argparse.ArgumentParser(description='Model validation script')
    parser.add_argument('--model-version', type=str, required=False, default="Staging")
    parser.add_argument('--data-rev', type=str, required=False, default="HEAD")
    parser.add_argument('--start-api', action='store_true')
    return parser.parse_args()

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("credit-card-fraud-detection")

def load_model(model_version: str):
    logger.info(f"Loading model version '{model_version}' from MLflow registry...")
    return mlflow.xgboost.load_model(f"models:/fraud-detection-model/{model_version}")

def load_test_data(data_rev: str):
    logger.info(f"Pulling test data from DVC revision: {data_rev}")
    subprocess.run(["dvc", "pull", "--force", PROCESSED_DATA_FILE_TEST.as_posix()], check=True)

    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }

    return metrics, y_pred, y_proba

def validate_performance(metrics: Dict[str, float]):
    failed_metrics = [
        key for key, val in PERFORMANCE_REQUIREMENTS.items()
        if metrics.get(key.replace("min_", ""), 0) < val
    ]

    for key, val in PERFORMANCE_REQUIREMENTS.items():
        print(key, val)
        print(metrics.get(key.replace("min_", ""), 0))
        
    if failed_metrics:
        logger.warning(f"Model failed to meet performance requirements for: {failed_metrics}")
        return False
    logger.info("Model passed all performance requirements.")
    return True

def create_visualizations(y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = VALIDATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = VALIDATION_DIR / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    return [str(cm_path), str(roc_path)]

def log_to_mlflow(metrics, artifacts, model_version, requirements_passed):
    mlflow.log_param("validated_model_version", model_version)
    mlflow.log_metrics(metrics)
    for artifact in artifacts:
        mlflow.log_artifact(artifact, artifact_path="validation")
    mlflow.set_tag("validation_passed", requirements_passed)

def setup_api(model):
    app = FastAPI()

    @app.get("/")
    def root():
        return {"message": "Fraud Detection Model Inference API"}

    @app.post("/predict")
    def predict(input_data: InferenceInput):
        try:
            X = pd.DataFrame([input_data.inputs])
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            return {"prediction": int(prediction), "probability": float(probability)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_args()
    setup_mlflow()

    with mlflow.start_run(run_name="validation"):
        model = load_model(args.model_version)
        X_test, y_test = load_test_data(args.data_rev)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        requirements_passed = validate_performance(metrics)
        artifact_paths = create_visualizations(y_test, y_pred, y_proba)
        log_to_mlflow(metrics, artifact_paths, args.model_version, requirements_passed)

        logger.info("Validation metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        if args.start_api:
            logger.info("Starting model inference API...")
            setup_api(model)

    logger.info("Model validation pipeline completed.")

if __name__ == "__main__":
    main()