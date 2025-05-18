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

## 🐳 Start MLflow + MinIO (Before Running Scripts)

You must start the MLflow Tracking Server and MinIO before running any pipeline scripts.
This is required for logging, model registration, and DVC remote storage.

Start them with Docker Compose:

```bash
docker-compose up -d
```

- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9000 (login using credentials from `.env`)

---

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

### Automatic Pipeline Execution on New Data

This pipeline is integrated with **GitHub Actions** to run automatically when new data is pushed to the repository under `data/raw/`.

Whenever new credit card data is added and pushed:

```bash
# Download or copy new data to data/raw/
python scripts/setup.py

# Re-track the data (if modified)
dvc add data/raw/creditcard-data.csv

# Commit and push to GitHub
git add data/raw/creditcard-data.csv.dvc
git commit -m "New data received"
git push
```

GitHub Actions will then:
- Automatically run `dvc repro` to execute the full pipeline
- Push updated artifacts to your DVC remote

> ✅ You no longer need to run scripts like `preprocess.py`, `train.py`, or `validate.py` manually — just `git push` your data update and everything runs automatically.

---

## 🔁 Manual Reproduction (Optional)

You can still manually run the pipeline locally using:

```bash
dvc repro
```

To view the pipeline graph:

```bash
dvc dag
```

To push outputs to remote:

```bash
dvc push
```

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
- Extend to cloud-native pipeline (e.g., with Airflow, SageMaker)
- Use asynchronous notifications when pipeline completes

---
