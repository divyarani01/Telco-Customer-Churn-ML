"""
Export the best model + preprocessing pipeline to models/ for Docker.

Run this from the project root after the notebook has completed:
    & "C:\ProgramData\anaconda3\python.exe" scripts\save_model_for_api.py
"""
import sys, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow
from config.config import MLFLOW_URI, REGISTRY_NAME, MODELS_DIR
from data.load_data import load_raw_data
from pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from sklearn.model_selection import train_test_split

MODELS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)

# ── Find the best run ─────────────────────────────────────────────────────────
client = mlflow.tracking.MlflowClient()

all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
if not all_versions:
    raise RuntimeError(f"No registered model '{REGISTRY_NAME}' found in {MLFLOW_URI}")

mv = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0]
run_id = mv.run_id
print(f"Best model : {REGISTRY_NAME} v{mv.version}  run_id={run_id}")

# ── Load with the correct MLflow flavour ─────────────────────────────────────
run = client.get_run(run_id)
model_type = run.data.params.get("model_type", "")
artifact_uri = run.info.artifact_uri
print(f"Model type : {model_type}")
print(f"Artifact   : {artifact_uri}")

model_uri = f"runs:/{run_id}/model"

def try_load(uri):
    loaders = [
        ("catboost",  lambda u: __import__("mlflow.catboost",  fromlist=["load_model"]).load_model(u)),
        ("xgboost",   lambda u: __import__("mlflow.xgboost",   fromlist=["load_model"]).load_model(u)),
        ("lightgbm",  lambda u: __import__("mlflow.lightgbm",  fromlist=["load_model"]).load_model(u)),
        ("sklearn",   lambda u: __import__("mlflow.sklearn",   fromlist=["load_model"]).load_model(u)),
        ("pyfunc",    lambda u: __import__("mlflow.pyfunc",    fromlist=["load_model"]).load_model(u)),
    ]
    for name, loader in loaders:
        try:
            m = loader(uri)
            print(f"Loaded via mlflow.{name}")
            return m
        except Exception as e:
            print(f"  mlflow.{name} failed: {e!s:.80}")
    raise RuntimeError("All MLflow loaders failed.")

model = try_load(model_uri)

# Unwrap pyfunc wrapper if needed
if hasattr(model, "_model_impl"):
    model = model._model_impl

model_path = MODELS_DIR / "best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Saved model          -> {model_path}")

# ── Build & fit preprocessing pipeline ───────────────────────────────────────
df = load_raw_data()
X  = df.drop(columns=["Churn"])
y  = df["Churn"].map({"Yes": 1, "No": 0})
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

is_linear = "LogisticRegression" in model_type
pipe = build_preprocessing_pipeline("linear" if is_linear else "tree")
pipe.fit(X_train, y_train)

pipe_path = MODELS_DIR / "preprocessing_pipe.pkl"
with open(pipe_path, "wb") as f:
    pickle.dump(pipe, f)
print(f"Saved preprocessing  -> {pipe_path}")

# ── Save threshold ────────────────────────────────────────────────────────────
threshold = float(run.data.params.get("best_threshold", 0.5))
(MODELS_DIR / "threshold.txt").write_text(str(threshold))
print(f"Saved threshold      -> {MODELS_DIR / 'threshold.txt'}  ({threshold})")

print("\nAll done. Now rebuild the Docker containers:")
print('  docker compose down && docker compose up --build')
