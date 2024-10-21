import kagglehub

# Download latest version
path = kagglehub.dataset_download("theprakharsrivastava/stargazer")

print("Path to dataset files:", path)