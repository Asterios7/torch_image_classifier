import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from util_functions import load_class_names
from torchClassifier import torchClassifier
import argparse

# 0. Create an argument parser
parser = argparse.ArgumentParser(description="Image classification prediction")
parser.add_argument("--image-path",
                    type=str,
                    default='data/oxford-iiit-pet/images/beagle_7.jpg',
                    help="path of the image to classify")
parser.add_argument("--mlflow-run",
                    type=str,
                    default='b12a9d24a0ce41a5b219fced426995a5',
                    help="MLflow run, retrieves model from this run")

args = parser.parse_args()

# 1. Set globals
IMAGE_PATH = args.image_path
MLFLOW_RUN = args.mlflow_run

# 2. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load the model from mlflow
model = mlflow.pytorch.load_model(f"runs:/{MLFLOW_RUN}/model").to(device)

# 4. Create data transform for image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 5. Load class_names
class_names, class_to_idx = load_class_names(
            class_names_path='./train_model/class_names.json',
            class_to_idx_path='./train_model/class_to_idx.json'
            )

# 6. Make prediction
classifier = torchClassifier()
print(classifier.predict(model=model,
                         image_path=IMAGE_PATH,
                         class_names=class_names,
                         image_size=(224, 224),
                         transform=transform,
                         device=device))
