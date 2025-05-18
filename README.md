# 💳 Credit Card Fraud Detection - MLOps Pipeline

This project implements a complete MLOps pipeline for detecting fraudulent credit card transactions using machine learning. It covers data versioning, preprocessing, model training, evaluation, tracking, and deployment, with reproducibility and scalability in mind.

---

## 🚀 Project Structure

```
.
├── data/                  # DVC-tracked datasets (raw, processed)
├── models/                # Trained models and validation outputs
├── scripts/               # Core scripts: setup.py, preprocess.py, train.py, validate.py
├── .dvc/                  # DVC configuration
├── dvc.yaml               # DVC pipeline definition
├── docker-compose.yml     # MLflow, MinIO setup
├── .env                   # Environment variables (AWS, DVC, MLflow)
├── README.md              # This documentation
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fraud-mlops-pipeline.git
cd fraud-mlops-pipeline
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 DVC Setup

### 1. Initialize DVC

```bash
dvc init
```

### 2. Configure Remote Storage (e.g., MinIO, S3)

```bash
dvc remote add -d myremote s3://mlops-bucket/data
dvc remote modify myremote endpointurl http://localhost:9000
dvc remote modify myremote access_key_id <your-access-key>
dvc remote modify myremote secret_access_key <your-secret-key>
```

> Your credentials can be stored in a `.env` file.

## ⚙️ Running the Pipeline

### Step 1: Download and Track Dataset

```bash
python scripts/setup.py
```

This downloads the dataset and adds it to DVC.

---

### Step 2: Preprocess Data

```bash
python scripts/preprocess.py --data-rev HEAD
```

This performs normalization, SMOTE, train/val/test split, and logs to MLflow.

---

### Step 3: Train Model

```bash
python scripts/train.py --data-rev HEAD
```

This trains an XGBoost model with hyperparameter tuning and registers it to MLflow.

---

### Step 4: Validate Model

```bash
python scripts/validate.py --model-version Staging --data-rev HEAD
```

This evaluates the model, checks performance thresholds, and logs metrics/artifacts.

To start the prediction API:

```bash
python scripts/validate.py --start-api
```

---

## 🔁 Automate Pipeline with DVC

Make sure your `dvc.yaml` file contains all pipeline stages.

Run the entire pipeline:

```bash
dvc repro
```

Visualize the pipeline graph:

```bash
dvc dag
```

Push results to remote storage:

```bash
dvc push
```

---

## 🐳 MLflow + MinIO via Docker

To start the tracking server and MinIO:

```bash
docker-compose up -d
```

- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9000 (login with keys in `.env`)

---

## 📊 MLflow Tracking

Tracked with:
- Parameters
- Metrics (accuracy, precision, recall, F1, ROC AUC)
- Artifacts (confusion matrix, ROC curve)
- Model versions (Staging, Production)

---

## 📈 Performance Requirements

Model must meet:

- Accuracy ≥ 0.98
- Precision ≥ 0.85
- Recall ≥ 0.70
- F1 Score ≥ 0.75
- ROC AUC ≥ 0.95

Validation fails if any are not met.

---

## ✅ Testing

Basic tests are run manually by executing:

- Each pipeline step individually with `--data-rev`
- `validate.py`'s API with sample input

Unit testing framework setup (e.g., `pytest`) is optional for future extension.

---

## 📎 Notes

- Environment variables must be set in `.env`
- Models are loaded via MLflow Registry (`models:/fraud-detection-model/Staging`)
- Data is versioned with DVC
- Works fully offline if DVC remote and MLflow tracking are self-hosted

---

## 🔧 Future Improvements

- Add `pytest`-based unit and integration tests
- Integrate CI/CD (e.g., GitHub Actions)
- Extend to cloud-native pipeline (e.g., with Airflow, SageMaker)

---

## 👨‍💻 Author

MLOps Pipeline by [Your Name]
