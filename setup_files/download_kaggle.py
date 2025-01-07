import kagglehub

# Download latest version
path = kagglehub.dataset_download("wangziang/waste-pictures")

print("Path to dataset files:", path)
