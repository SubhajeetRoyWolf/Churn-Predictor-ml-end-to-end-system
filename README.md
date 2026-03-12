# Olist Customer Churn Prediction System
An end-to-end Machine Learning pipeline to predict customer churn risk (Low Review Scores) using order logistics and pricing data.

---

## 📸 Dashboard Overview
> <img width="955" height="401" alt="image" src="https://github.com/user-attachments/assets/07f8fdb9-f824-48a2-b6bf-8d013612eb8d" />

---

## 📁 Project Architecture
The system is built using a microservices architecture, containerized with Docker:

* **`app/`**: FastAPI backend logic and prediction scripts.
* **`model/`**: Storage for the serialized Random Forest model (`olist_pipeline.pkl`).
* **`src/`**: Training pipeline and data processing scripts.
* **`data/sample_data.csv`**: The dataset used for training and validation.
* **`mlflow.db`**: SQLite database tracking experiment history and model versions.
* **`docker-compose.yml`**: Orchestration file to launch the API and UI together.

---

## 🧠 Business Logic & Model
The model predicts the probability of a **"Low Review"** (Review Score ≤ 2), which serves as a proxy for customer churn.

* **Algorithm**: Random Forest Classifier.
* **Features**: `price`, `freight_value`, `delivery_time`.
* **Tracking**: All hyperparameters and metrics are logged in MLflow for full reproducibility.


---

## Features
Machine Learning pipeline
Hyperparameter tuning
Model explainability (SHAP)
FastAPI prediction API
Streamlit dashboard
MLflow experiment tracking
Docker containerization

## Tech Stack
Python
Scikit-learn
FastAPI
Streamlit
MLflow
Docker


## Architecture
User → Streamlit UI → FastAPI → ML Model → Prediction

## 🚀 Deployment (Quick Start)
This project is fully containerized. No local Python installation is required.

### 1. Build and Launch
Navigate to the project root in your terminal and run:
```bash
docker-compose up --build

Note : 
🛠️ Training the Model
If it need to retrain the model locally:
Ensure sample.csv is in the root directory.

Run the training script:
Bash
python Churn-Predictor-ml-end-to-end-system/src/train_pipeline.py
