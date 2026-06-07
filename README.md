# Workflow CI - Customer Churn Prediction

## Deskripsi Proyek

Proyek ini merupakan implementasi Continuous Integration (CI) untuk pelatihan ulang (retraining) model Machine Learning menggunakan MLflow Project dan GitHub Actions.

Dataset yang digunakan adalah IBM Telco Customer Churn Dataset yang telah melalui tahap preprocessing pada repository eksperimen sebelumnya.

## Struktur Proyek
Workflow-CI/
│
├── .github/
│   └── workflows/
│       └── retrain.yml
│
└── MLProject/
    ├── MLproject
    ├── conda.yaml
    ├── modelling.py
    ├── requirements.txt
    ├── Dockerfile
    ├── DockerHub.txt
    └── dataset_preprocessing/
            |__telco_churn_processed.csv

## Menjalankan MLflow Project
cd MLProject

mlflow run . --env-manager=local

## Menjalankan Docker Image

Build image:
docker build -t telco-churn-mlflow .

Run container:
docker run --rm telco-churn-mlflow

## Docker Hub Repository
Docker image tersedia pada:
https://hub.docker.com/r/mfsirojudin/telco-churn-mlflow

## Workflow CI
GitHub Actions akan otomatis melakukan retraining model setiap kali terdapat perubahan pada branch `main`.
