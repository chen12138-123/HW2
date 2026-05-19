""" Download dataset from Kaggle  """
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashfakyeafi/road-vehicle-images-dataset", output_dir="MLLM/data")

print("Path to dataset files:", path)