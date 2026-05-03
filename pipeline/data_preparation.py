
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# Credentials loaded from environment
HF_USERNAME  = "harshverma27"
HF_TOKEN     = os.environ.get("HF_TOKEN")
dataset_repo = f"{HF_USERNAME}/tourism-dataset"

# Pull raw CSV from HF Hub
raw_path = hf_hub_download(
    repo_id=dataset_repo,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN
)

df = pd.read_csv(raw_path)
