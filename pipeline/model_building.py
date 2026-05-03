
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from huggingface_hub import HfApi, hf_hub_download

# Credentials loaded from environment
HF_USERNAME  = "harshverma27"
HF_TOKEN     = os.environ.get("HF_TOKEN")
dataset_repo = f"{HF_USERNAME}/tourism-dataset"
model_repo   = f"{HF_USERNAME}/tourism-wellness-model"

# MLflow runs locally on the Actions runner
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism_wellness_prediction")

# Pull prepared splits from HF
train_path = hf_hub_download(repo_id=dataset_repo, filename="train.csv",
                              repo_type="dataset", token=HF_TOKEN)
test_path  = hf_hub_download(repo_id=dataset_repo, filename="test.csv",
                              repo_type="dataset", token=HF_TOKEN)

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]
X_test  = test_df.drop(columns=["ProdTaken"])
y_test  = test_df["ProdTaken"]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# 6 candidate models with hyperparameter grids
models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 7], "min_samples_split": [2, 5, 10]}
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"n_estimators": [10, 50, 100]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1.0]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    ),
}

best_f1    = 0
best_model = None
best_name  = ""

# Train, tune and log each model
for name, (model, params) in models.items():
    print(f"Training {name}...")
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(model, params, cv=3, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)

        preds = gs.best_estimator_.predict(X_test)
        f1    = f1_score(y_test, preds)
        auc   = roc_auc_score(y_test, gs.best_estimator_.predict_proba(X_test)[:, 1])

        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics({"f1": f1, "auc": auc})
        mlflow.sklearn.log_model(gs.best_estimator_, name)

        print(f"  F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_model = gs.best_estimator_
            best_name  = name

print(f"\n** Best model: {best_name} | F1: {best_f1:.4f}")

# Save and push best model to HF Hub
os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, "tourism_project/model_building/best_model.pkl")

api = HfApi()
api.upload_file(
    path_or_fileobj="tourism_project/model_building/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=model_repo,
    repo_type="model",
    token=HF_TOKEN
)
print(f" Best model pushed -> https://huggingface.co/models/{model_repo}")
