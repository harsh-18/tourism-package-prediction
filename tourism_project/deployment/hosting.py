
import os
from huggingface_hub import HfApi

# -> credentials and space details
HF_USERNAME = "harshverma27"
HF_TOKEN    = os.environ.get("HF_TOKEN")
space_repo  = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi()

# -> create the HF Space if it doesn't already exist
# -- sdk is set to docker since we are using our own Dockerfile
api.create_repo(
    repo_id=space_repo,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    token=HF_TOKEN
)

# -> push all three deployment files into the Space repo
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    api.upload_file(
        path_or_fileobj=f"/content/drive/MyDrive/Colab Notebooks/PGP AI ML/Projects/tourism_project/deployment/{fname}",
        path_in_repo=fname,
        repo_id=space_repo,
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"{fname} uploaded")

print(f"App will be live at -> https://huggingface.co/spaces/{space_repo}")
