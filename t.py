from huggingface_hub import HfFolder, snapshot_download

# Get the expected download location for the specific dataset
dataset_name = "biglab/webui-7k"
expected_download_path = snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",
    ignore_patterns=["*"],
    max_workers=1,
)

print(f"The dataset '{dataset_name}' would be downloaded to:")
print(expected_download_path)
