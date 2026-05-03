
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# -> credentials loaded from environment -- set via GitHub Actions secrets
HF_USERNAME  = "harshverma27"
HF_TOKEN     = os.environ.get("HF_TOKEN")
dataset_repo = f"{HF_USERNAME}/tourism-dataset"

# -> pull raw CSV from HF Hub
raw_path = hf_hub_download(
    repo_id=dataset_repo,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN
)

df = pd.read_csv(raw_path)
print(f"Loaded {df.shape[0]} rows")

# -> drop identifier column, not useful for modelling
df.drop(columns=["CustomerID"], inplace=True)

# -> fix inconsistent values in the raw data
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")
df["Gender"]        = df["Gender"].replace("Fe Male", "Female")

# -> fill missing numeric values with column median
num_cols = ["Age", "NumberOfTrips", "NumberOfChildrenVisiting",
            "MonthlyIncome", "DurationOfPitch"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df.dropna(inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# -> encode categorical columns into numbers using label encoding
cat_cols = ["TypeofContact", "Occupation", "Gender",
            "MaritalStatus", "Designation", "ProductPitched"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# -> separate features from target, scale features
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -> 80/20 stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy(); train_df["ProdTaken"] = y_train.values
test_df  = X_test.copy();  test_df["ProdTaken"]  = y_test.values

# -> save splits locally then push both to HF
os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv",   index=False)
print(f"Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows")

api = HfApi()
for fname in ["train.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{fname}",
        path_in_repo=fname,
        repo_id=dataset_repo,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"{fname} uploaded to HF")
