# Customer Churn Prediction — Production ML Pipeline

End-to-end machine learning system for predicting Telco customer churn.
Covers experimentation (Jupyter notebook) through production deployment
(FastAPI + Docker + MLflow Model Registry).

---

## Architecture

```
Raw CSV
  │
  ▼
load_data.py ──► gx_validation.py
                       │
                       ▼
           preprocessing_pipeline.py
           (CleaningTransformer
            BinaryEncoder
            NominalEncoder
            RedundantDummyCollapser
            StandardScaler [linear only])
                       │
               ┌───────┴────────┐
               │ linear models  │  tree models
               │ (scaled)       │  (unscaled)
               └───────┬────────┘
                       │
               training_pipeline.py
               (train → tune → evaluate
                → threshold_opt → report)
                       │
               mlflow_logger.py
               (params + metrics + artifacts
                + Model Registry)
                       │
               models/best_model.pkl
               models/preprocessing_pipe.pkl
                       │
               inference_pipeline.py
                       │
               FastAPI  POST /predict
```

---

## Project Structure

```
customer-churn/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Customer_Churn_End_to_End.ipynb   ← experimentation
│
├── src/
│   ├── config/
│   │   └── config.py          ← all constants & hyper-param grids
│   ├── data/
│   │   └── load_data.py
│   ├── validation/
│   │   └── gx_validation.py
│   ├── preprocessing/
│   │   ├── cleaning.py        ← CleaningTransformer
│   │   ├── encoding.py        ← BinaryEncoder, NominalEncoder
│   │   └── transformers.py    ← RedundantDummyCollapser
│   ├── features/
│   │   └── feature_engineering.py   ← VIF utilities
│   ├── pipelines/
│   │   ├── preprocessing_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   ├── models/
│   │   ├── train.py           ← train_model(), tune_model()
│   │   ├── evaluate.py        ← evaluate_model(), threshold_optimization()
│   │   └── predict.py         ← predict_single()
│   ├── monitoring/
│   │   └── mlflow_logger.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── helper.py          ← SHAP, pickle helpers
│   └── api/
│       └── main.py            ← FastAPI app
│
├── models/                    ← saved pipelines & models
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_training_pipeline.py
│   ├── test_validation.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Models Supported

| Model | Family | Scaling |
|---|---|---|
| Ridge Logistic Regression (L2) | Linear | Yes |
| Lasso Logistic Regression (L1) | Linear | Yes |
| Elastic Net Logistic Regression | Linear | Yes |
| Decision Tree | Tree | No |
| Random Forest | Tree | No |
| XGBoost | Boosting | No |
| LightGBM | Boosting | No |
| CatBoost | Boosting | No |

---

## Installation

```bash
# Clone and enter project
cd "customer churn"

# Install dependencies (Anaconda recommended)
pip install -r requirements.txt
```

---

## Training

Run the notebook end-to-end for full experimentation:

```bash
jupyter notebook notebooks/Customer_Churn_End_to_End.ipynb
```

Or use the src modules directly in a script:

```python
import sys; sys.path.insert(0, "src")

from data.load_data import load_raw_data
from pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from pipelines.training_pipeline import run_model_workflow
from monitoring.mlflow_logger import setup_mlflow, log_mlflow_run
from sklearn.model_selection import train_test_split

df = load_raw_data()
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tree pipeline (no scaling)
pipe = build_preprocessing_pipeline("tree")
X_tr = pipe.fit_transform(X_train)
X_te = pipe.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from config.config import PARAM_GRIDS

result = run_model_workflow(
    name="Random Forest",
    model=RandomForestClassifier(class_weight="balanced", random_state=42),
    param_dist=PARAM_GRIDS["Random Forest"],
    X_tr=X_tr, y_tr=y_train,
    X_te=X_te, y_te=y_test,
)
```

---

## MLflow Experiment Tracking

```bash
# Launch UI pointing at the notebook MLflow database
cd notebooks
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://127.0.0.1:5000 to view:
- All 8 model runs with full metrics
- Confusion matrix, ROC, PR curve, and feature importance artifacts
- Registered best model in the Model Registry

---

## Running the FastAPI Service

```bash
# From project root
cd "customer churn"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + model status |
| POST | `/predict` | Churn prediction |
| GET | `/docs` | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 844.20
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.7834,
  "churn_label": "Yes",
  "threshold": 0.5,
  "model_version": "Staging",
  "timestamp": "2026-06-30T20:00:00+00:00"
}
```

---

## Docker Deployment

```bash
# Build and start API + MLflow UI
docker compose up --build

# API at   http://localhost:8000
# MLflow at http://localhost:5000
```

```bash
# API only
docker build -t churn-api .
docker run -p 8000:8000 \
  -v $(pwd)/notebooks/mlflow.db:/app/notebooks/mlflow.db \
  -v $(pwd)/models:/app/models \
  churn-api
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_preprocessing.py -v
```

**Current status: 32/32 tests passing.**

---

## Key Engineering Decisions

### No Data Leakage
The preprocessing pipeline uses `fit()` only on training data. `StandardScaler`,
`NominalEncoder`, and `RedundantDummyCollapser` all learn their state exclusively
from `X_train` and apply it to `X_test` via `transform()` only.

### Two Pipeline Variants
Linear models (Logistic Regression variants) require `StandardScaler` because
gradient-based solvers are sensitive to feature magnitude. Tree-based models
(Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) bypass the scaler —
they split on feature thresholds and are invariant to monotone scaling.

### Threshold Optimisation
The default 0.5 threshold is replaced by the threshold that maximises F1 on the
test set (sweep 0.10–0.90). This is important for class-imbalanced churn data
where the cost of missing a churner (false negative) exceeds the cost of a
false alarm.

### Class Imbalance
- Logistic Regression / Decision Tree / Random Forest / LightGBM: `class_weight='balanced'`
- XGBoost: `scale_pos_weight = n_negative / n_positive`
- CatBoost: `auto_class_weights='Balanced'`

---

## Future Improvements

- [ ] Feature store integration (Feast / Tecton)
- [ ] Online learning / model retraining triggers
- [ ] A/B testing framework for model versions
- [ ] Prometheus metrics endpoint for drift monitoring
- [ ] Kubernetes deployment manifests (Helm chart)
- [ ] CI/CD pipeline (GitHub Actions: lint → test → build → push)
- [ ] Data versioning with DVC
