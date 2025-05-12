# Credit Card Fraud Detection MLOps Assessment

## Overview

In this technical assessment, you will build a complete MLOps pipeline for credit card fraud detection. This task is designed to evaluate your ability to implement MLOps best practices across the entire machine learning lifecycle, from data versioning to model deployment.

## Background

Credit card fraud detection is a real-world problem where machine learning can be highly effective. The task involves identifying fraudulent transactions from normal ones in a highly imbalanced dataset. Your challenge is to build not just a model, but a production-ready MLOps pipeline that addresses the entire workflow.

## Dataset

You will be working with the Credit Card Fraud Detection dataset, which contains transactions made by credit cards. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) accounts for 0.172% of all transactions.

The dataset has been anonymized and contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, ... V28 are the principal components obtained with PCA. The only features which have not been transformed with PCA are 'Time' and 'Amount'.

## Tasks

### 1. Infrastructure Setup
Implement a `docker-compose.yml` file that sets up:
- MLflow tracking server with a database backend
- DVC remote storage (using MinIO or similar)
- Any additional services you consider necessary

### 2. Data Acquisition
Complete the `setup.py` script to:
- Download the Credit Card Fraud Detection dataset from the provided URL
- Initialize DVC tracking for the dataset
- Push the raw data to DVC storage

### 3. Data Preprocessing
Develop the `preprocess.py` script to:
- Load a specific version of the raw data from DVC (version passed via command line)
- Handle class imbalance (using techniques like SMOTE, undersampling, etc.)
- Normalize/scale features
- Split data into training, validation, and test sets
- Save the processed datasets back to DVC with appropriate versioning
- Log preprocessing metrics and parameters to MLflow

### 4. Model Training
Implement the `train.py` script to:
- Load preprocessed data from a specific DVC version
- Train a Gradient Boosting model (XGBoost, LightGBM, etc.)
- Implement hyperparameter tuning
- Track experiments with MLflow
- Register the best model in the MLflow model registry
- Save model artifacts with proper versioning

### 5. Model Evaluation
Create the `validate.py` script to:
- Load a trained model from MLflow
- Evaluate on test data
- Calculate performance metrics (precision, recall, F1, AUC, etc.)
- Run validation checks against performance requirements
- Create evaluation visualizations
- Set up a simple API endpoint for model inference

## Requirements

Your solution should demonstrate:

1. **Reproducibility**: Someone else should be able to run your pipeline and get the same results
2. **Version Control**: Proper versioning of data, code, and models
3. **Experiment Tracking**: Comprehensive logging of metrics, parameters, and artifacts
4. **Pipeline Automation**: Clear organization of the ML workflow
5. **Error Handling**: Robust handling of errors and edge cases
6. **Documentation**: Clear documentation of your solution
7. **Testing**: Basic tests to ensure functionality

## Evaluation Criteria

Your submission will be evaluated on:
- Code quality and organization
- MLOps best practices implementation
- End-to-end functionality
- Documentation quality
- Error handling and robustness

## Deliverables

1. Complete the provided skeleton files from this repository
2. Add any additional files you consider necessary
3. Create a README.md file with:
   - Setup instructions
   - How to run each component
   - Any additional notes or comments

## Time Expectation

This assessment is designed to take approximately 4-8 hours to complete. It's not expected that you implement every possible MLOps feature, but rather demonstrate your understanding of key MLOps concepts and your ability to implement them.

Good luck!
