
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
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from huggingface_hub import HfApi, hf_hub_download

# -> credentials loaded from environment -- set via GitHub Actions secrets
HF_USERNAME  = "harshverma27"
HF_TOKEN     = os.environ.get("HF_TOKEN")
dataset_repo = f"{HF_USERNAME}/tourism-dataset"
model_repo   = f"{HF_USERNAME}/tourism-wellness-model"

# -> no ngrok in CI -- MLflow runs locally on the Actions runner
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_wellness_prediction")

# -> pull prepared splits from HF
train_path = hf_hub_download(repo_id=dataset_repo, filename="train.csv",
                              repo_type="dataset", token=HF_TOKEN)
test_path  = hf_hub_download(repo_id=dataset_repo, filename="test.csv",
                              repo_type="dataset", token=HF_TOKEN)

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# drop index column if present from old CSV saves
for _df in [train_df, test_df]:
    if "Unnamed: 0" in _df.columns:
        _df.drop(columns=["Unnamed: 0"], inplace=True)

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]
X_test  = test_df.drop(columns=["ProdTaken"])
y_test  = test_df["ProdTaken"]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# -> 6 candidate models with hyperparameter grids for tuning
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

# -> train, tune and log each model -- best F1 
