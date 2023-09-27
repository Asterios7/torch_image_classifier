# Imports
import torch
from torchvision import transforms
from torchmetrics import Accuracy
from torchinfo import summary
from torchClassifier import torchClassifier
import argparse
from datetime import date
from util_functions import (load_data,
                            create_dataloaders,
                            create_effnetb0_model,
                            save_model)

# 0. Create an argument parser
parser = argparse.ArgumentParser(description="Torch classification training")
parser.add_argument("--batch-size",
                    type=int, default=32,
                    help="Batch size")
parser.add_argument("--learning-rate",
                    type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("--epochs",
                    type=int, default=5,
                    help="Number of epochs")
parser.add_argument("--mlflow-experiment",
                    type=str, default="torch_test",
                    help="MLflow experiment name")
parser.add_argument("--mlflow-run",
                    type=str, default="Wicked_Jonathan",
                    help="MLflow experiment run name")
parser.add_argument("--display-summary",
                    type=bool, default=False,
                    help="Display model summary True/False")
args = parser.parse_args()

# 1. Set globals
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
MLFLOW_EXPERIMENT = args.mlflow_experiment
MLFLOW_EXPERIMENT_RUN = args.mlflow_run
DISPLAY_SUMMARY = args.display_summary

# 2. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# 3. Create EffnetB0 model data transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. Load train and test datasets
train_dataset, test_dataset = load_data(root="data",
                                        train_transforms=train_transforms,
                                        test_transforms=test_transforms)
class_names = train_dataset.classes
class_names_idx = train_dataset.class_to_idx
print(f"[INFO] Train dataset number of samples: {len(train_dataset)}")
print(f"[INFO] Test dataset number of samples: {len(test_dataset)}")

# 5. Create dataloaders
train_dataloader, test_dataloader = create_dataloaders(
                                        train_dataset=train_dataset,
                                        test_dataset=test_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=2
                                        )
print(f"[INFO] Train DataLoader number of batches: {len(train_dataloader)}")
print(f"[INFO] Test DataLoader number of batches: {len(test_dataloader)}")
# Batch shape (batch_size, color_channels, height, width)
print(f"[INFO] Batch shape: {next(iter(train_dataloader))[0].shape}")

# 6. Create a model instance
effnetb0 = create_effnetb0_model(num_classes=(len(class_names)),
                                 seed=13)
if DISPLAY_SUMMARY:
    summary(effnetb0,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

# 7. Select Optimizer, loss function and metric
optimizer = torch.optim.Adam(params=effnetb0.parameters(),
                             lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task='multiclass',
                       num_classes=len(class_names)).to(device)

# 8. Instantiate torchClassifier and fit model
classifier = torchClassifier()
print("[INFO] Starting training...")
torch.manual_seed(13)
results = classifier.fit(model=effnetb0,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         optimizer=optimizer,
                         loss_fn=loss_fn,
                         accuracy_fn=accuracy_fn,
                         epochs=EPOCHS,
                         device=device,
                         mlflow_experiment=MLFLOW_EXPERIMENT,
                         mlfow_run_name=MLFLOW_EXPERIMENT_RUN)

# 9. Save model in a models folder
today = date.today()
save_model(model=effnetb0,
           target_dir='models',
           model_name=f"""effnetb0_{today}_test-acc-{
            int(results['test_acc'][-1]*1e+4)}.pt""")
