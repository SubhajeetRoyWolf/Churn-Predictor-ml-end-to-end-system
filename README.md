# Customer Churn Prediction — Production ML Pipeline (Olist)

End-to-end customer churn prediction on the Olist Brazilian e-commerce dataset,
deployed as a containerized FastAPI microservice with a Streamlit UI.

## Stack

- **RandomForestClassifier** with `class_weight='balanced'` (handles ~15% positive class)
- **Feature engineering** — 7 features: price, freight_value, delivery_time, total_cost, freight_ratio, is_expensive, is_delayed
- **Hyperparameter tuning** via `RandomizedSearchCV` (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- **SHAP explainability** (TreeExplainer) — feature importance + beeswarm plots
- **MLflow experiment tracking** — params, metrics, model, artifacts logged to SQLite store
- **Full evaluation** on stratified hold-out: ROC-AUC, accuracy, precision, recall, F1, confusion matrix
- **FastAPI** microservice (`/predict`), Pydantic-validated inputs
- **Streamlit dashboard** with risk-band UI
- **Docker Compose** for the full stack

## Results (on full Olist dataset, ~115K rows)

| Metric            | Value |
| ----------------- | ----- |
| ROC-AUC (baseline RF) | **0.72** |
| ROC-AUC (tuned RF)    | **0.77** |
| Accuracy              | 0.86 |

> The included `data/sample_data.csv` is a 1K-row sample for CI/demo purposes — on this sample,
> AUC will be much lower (~0.50–0.65) due to the small size and severe class imbalance.
> Reproduce the headline numbers above by training on the full cleaned Olist dataset.

## Project Structure

```
churn/
├── app/
│   ├── main.py        # FastAPI app
│   ├── predictor.py   # Inference: builds engineered features, predicts
│   ├── schema.py      # Pydantic input
│   └── logger.py
├── src/
│   └── train_pipeline.py   # RF + engineered features + SHAP + MLflow + RandomizedSearchCV
├── data/sample_data.csv    # 1K demo sample
├── model/olist_pipeline.pkl  # Trained model + feature order
├── artifacts/                # SHAP plots, confusion matrix
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run

```bash
# Quick train with notebook-best params (default)
python src/train_pipeline.py

# Full hyperparameter search (slow, full dataset)
DO_TUNE=1 python src/train_pipeline.py

# Serve API
uvicorn app.main:app --reload

# View MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Full stack (API + UI + MLflow)
docker-compose up
```

## API

- `GET  /`         — health check
- `POST /predict`  — input: `{price, freight_value, delivery_time, is_delayed?}` → `{prediction, probability}`

## Roadmap

- Add review-comment text features (sentiment, length, presence)
- Move from local SQLite MLflow store to remote tracking server
- CI gate on ROC-AUC threshold
- Compare RF vs GradientBoosting vs XGBoost as part of evaluation suite
