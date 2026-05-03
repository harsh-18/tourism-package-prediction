
import os
from huggingface_hub import HfApi

# Credentials and repo details loaded from environment
HF_USERNAME  = "harshverma27"
HF_TOKEN     = os.environ.get("HF_TOKEN")
dataset_repo = f"{HF_USERNAME}/tourism-dataset"

api = HfApi()

# Create the dataset repo on HF if it doesn't already exist
api.create_repo(
    repo_id=dataset_repo,
    repo_type="dataset",
    exist_ok=True,
    token=HF_TOKEN
)

# Upload the raw CSV -- GitHub Actions will have this file via checkout
api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=dataset_repo,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"Dataset uploaded -> https://huggingface.co/datasets/{dataset_repo}")
