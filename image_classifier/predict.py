import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from torchClassifier import torchClassifier
import argparse
from util_functions import (load_class_names,
                            create_effnetb0_model)

# 0. Create an argument parser
parser = argparse.ArgumentParser(description="Image classification prediction")
parser.add_argument("--image-path",
                    type=str,
                    default='data/oxford-iiit-pet/images/beagle_7.jpg',
                    help="path of the image to classify")
parser.add_argument("--mlflow-run",
                    type=str,
                    default=None,
                    help="MLflow run, retrieves model from this run")
parser.add_argument("--model-path",
                    type=str,
                    default="models/effnetb0_2023-09-27_test-acc-8555.pt",
                    help="Path of the model to use for predictions")

args = parser.parse_args()

# 1. Set globals
IMAGE_PATH = args.image_path
MLFLOW_RUN = args.mlflow_run
MODEL_PATH = args.model_path

# 2. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load class_names
class_names, class_to_idx = load_class_names(
            class_names_path='./image_classifier/class_names.json',
            class_to_idx_path='./image_classifier/class_to_idx.json'
            )

# 4. Load the model (if MLflow run is not given load from models dir)
if MLFLOW_RUN is not None:
    model = mlflow.pytorch.load_model(f"runs:/{MLFLOW_RUN}/model").to(device)
else:
    model = create_effnetb0_model(num_classes=len(class_names))
    model.load_state_dict(
        torch.load(f=MODEL_PATH,
                   map_location=torch.device(device)))

# 5. Create data transform for image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 6. Make prediction
classifier = torchClassifier()
print(classifier.predict(model=model,
                         image_path=IMAGE_PATH,
                         class_names=class_names,
                         image_size=(224, 224),
                         transform=transform,
                         device=device))
