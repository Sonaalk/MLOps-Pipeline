❤️ Heart Disease Prediction – End-to-End MLOps Pipeline
📌 Project Overview

This repository contains an end-to-end MLOps implementation for predicting the presence of heart disease using patient health data from the UCI Heart Disease dataset.

The project demonstrates how a machine learning model can be taken from data ingestion to production deployment, following modern MLOps best practices such as automation, reproducibility, CI/CD, containerization, and monitoring.

🎯 Objective

To build a scalable and reproducible machine learning system that:

Predicts the risk of heart disease (binary classification)

Exposes predictions via a REST API

Can be deployed in a containerized and cloud-ready environment

Includes automated testing and CI/CD workflows

📊 Dataset

Name: Heart Disease Dataset

Source: UCI Machine Learning Repository

Access Method: ucimlrepo Python package

Features: Age, sex, chest pain type, blood pressure, cholesterol, heart rate, etc.

Target Variable:

0 → No heart disease

1 → Presence of heart disease

✔ Dataset is fetched programmatically to ensure authenticity and reproducibility.

🧱 Project Structure
ML OPS/
│── heart_disease_mlops.ipynb     # Single notebook: EDA + training + MLflow
│── train.py                      # Training script for CI/CD
│── app.py                        # FastAPI model serving app
│── requirements.txt              # Project dependencies
│── Dockerfile                    # Docker image definition
│── data/
│   └── heart.csv                 # Processed dataset
│── models/
│   └── final_model/              # Saved MLflow model
│── tests/
│   ├── test_data.py              # Data validation tests
│   └── test_model.py             # Model tests
│── k8s/
│   ├── deployment.yaml           # Kubernetes deployment
│   └── service.yaml              # Kubernetes service
│── .github/workflows/
│   └── mlops.yml                 # GitHub Actions CI/CD pipeline
│── README.md

🔬 Exploratory Data Analysis (EDA)

EDA is performed in a single Jupyter Notebook and includes:

Dataset shape and schema inspection

Missing value checks

Class distribution visualization

Feature distribution plots

Correlation heatmap

All outputs are generated interactively within the notebook.

🤖 Model Development

Algorithms Used:

Logistic Regression

Random Forest Classifier

Preprocessing Steps:

Median imputation for missing values

Feature scaling using StandardScaler

Evaluation Strategy:

5-fold cross-validation

ROC-AUC as the primary metric

Final Model Selected: Random Forest (based on performance)

📈 Experiment Tracking (MLflow)

MLflow is used to:

Track experiments and metrics

Log and version trained models

Compare multiple runs

Store artifacts reproducibly

Run locally:

mlflow ui

🧪 Automated Testing

Unit tests are written using Pytest to validate:

Dataset availability and integrity

Target column correctness

Model loading

Model prediction functionality

Run tests:

python -m pytest -v

🔄 CI/CD Pipeline

A GitHub Actions workflow automates:

Dependency installation

Unit test execution

Model training script execution

Failure on test or code errors

✔ Ensures production readiness and reproducibility

🌐 Model Serving (FastAPI)

The trained model is exposed via a FastAPI REST API.

Run locally:
uvicorn app:app --reload

API Endpoint:

POST /predict → Returns prediction and confidence score

Swagger UI:

http://127.0.0.1:8000/docs

🐳 Docker Containerization

The FastAPI application is containerized using Docker.

docker build -t heart-api .
docker run -p 8000:8000 heart-api

☸️ Kubernetes Deployment

The Dockerized API is deployed using Kubernetes (Minikube / Docker Desktop).

Components:

Deployment (replicas & container config)

Service (LoadBalancer)

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

📊 Monitoring & Logging

API request logging using Python logging

Logs accessible via:

docker logs

kubectl logs

Kubernetes pod & service health used as basic monitoring

🛠️ Installation & Setup
pip install -r requirements.txt
python train.py

📌 Key MLOps Practices Demonstrated

✔ Reproducible data ingestion

✔ Pipeline-based preprocessing

✔ Experiment tracking

✔ Automated testing

✔ CI/CD automation

✔ Containerized model serving

✔ Kubernetes deployment

✔ Logging and monitoring

📄 License / Academic Use

This project is developed for academic and learning purposes as part of an MLOps experimental learning assignment.
