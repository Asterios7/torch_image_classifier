import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import nn
import torchvision
from torchvision import datasets
import os
from torch.utils.data import DataLoader
from typing import Tuple
import json


def plot_loss_curves(results: dict) -> None:
    """Plots training curves of a results dictionary using Plotly.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    epochs = list(range(len(results["train_loss"])))

    # Create loss figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=results["train_loss"],
                                  mode="lines", name="train_loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=results["test_loss"],
                                  mode="lines", name="test_loss"))
    loss_fig.update_layout(title="Loss", xaxis_title="Epochs",
                           yaxis_title="Loss")

    # Create accuracy figure
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epochs, y=results["train_acc"],
                                      mode="lines", name="train_accuracy"))
    accuracy_fig.add_trace(go.Scatter(x=epochs, y=results["test_acc"],
                                      mode="lines", name="test_accuracy"))
    accuracy_fig.update_layout(title="Accuracy", xaxis_title="Epochs",
                               yaxis_title="Accuracy")

    # Create subplots
    subplot_titles = ("Loss", "Accuracy")
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)
    fig.add_trace(loss_fig.data[0], row=1, col=1)
    fig.add_trace(loss_fig.data[1], row=1, col=1)
    fig.add_trace(accuracy_fig.data[0], row=1, col=2)
    fig.add_trace(accuracy_fig.data[1], row=1, col=2)

    # Update layout
    fig.update_layout(title_text="Training Curves")
    fig.show()


def create_effnetb0_model(num_classes: int = 3,
                          seed: int = 13) -> torch.nn.Module:
    """
    Create and configure an EfficientNet-B0 model for a custom number
    of output classes.

    Args:
        num_classes (int, optional): Number of output classes for
            classification. Default is 3.
        seed (int, optional): Random seed for reproducibility. Default is 13.

    Returns:
        torch.nn.Module: A pre-trained EfficientNet-B0 model with
        the specified number of output classes and a modified
        classifier head.

    This function creates an EfficientNet-B0 model from torchvision and
    configures it for a specific classification task. It sets the model's
    classifier head to have the desired number of output classes and applies
    dropout for regularization. The function also allows for setting a random
    seed to ensure reproducible results.
    """
    # Get weights and model
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return model


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or \
           model_name.endswith(".pt"), "model_name must end with '.pt','.pth'"
    model_save_path = os.path.join(target_dir, model_name)

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model: torch.nn.Module,
               model_path: str) -> torch.nn.Module:
    """Loads a PyTorch model from a specified model path.

    Args:
    model: An instance of the PyTorch model to load the state_dict into.
    model_path: The path to the saved model file.

    Example usage:
    loaded_model = load_model(model=my_model, model_path="models/my_model.pth")
    """
    # Load the model state_dict
    model.load_state_dict(torch.load(model_path))

    # Ensure the model is in evaluation mode
    model.eval()

    print(f"[INFO] Loaded model from: {model_path}")

    return model


def load_data(root: str,
              train_transforms: torchvision.transforms,
              test_transforms: torchvision.transforms) -> tuple:

    # Load train and test datasets
    train_dataset = datasets.OxfordIIITPet(root=root,
                                           split="trainval",
                                           transform=train_transforms,
                                           download=True)

    test_dataset = datasets.OxfordIIITPet(root=root,
                                          split="test",
                                          transform=test_transforms,
                                          download=True)

    return train_dataset, test_dataset


def create_dataloaders(train_dataset: torchvision.datasets,
                       test_dataset: torchvision.datasets,
                       batch_size: int = 32,
                       num_workers: int = 2) -> tuple:

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    return train_dataloader, test_dataloader


def load_class_names(class_names_path: str,
                     class_to_idx_path: str
                     ) -> Tuple[list, dict]:
    """
    Load class names and class-to-index mapping from JSON files.

    Args:
        class_names_path (str): Path to a JSON file containing class names.
        class_to_idx_path (str): Path to a JSON file containing class-to-index
        mapping.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing two elements:
            - A list of class names.
            - A dictionary mapping class names to their corresponding indices.

    Raises:
        FileNotFoundError: If either of the specified JSON files does
        not exist.
        KeyError: If the 'class_names' key is not found in the class names
        JSON file.
        JSONDecodeError: If there is an issue decoding the JSON content.
        AssertionError: If the input file paths do not have the expected
        '.json' file extension.

    Example:
        class_names, class_to_idx = load_class_names('class_names.json',
                                                     'class_to_idx.json')
    """
    assert class_names_path.endswith(".json"),\
        "class_names_path should end with class_names.json"
    assert class_to_idx_path.endswith(".json"),\
        "class_to_idx_path should end with class_to_idx.json"

    with open(class_names_path, 'r') as fp:
        class_names = json.load(fp)['class_names']

    with open(class_to_idx_path, 'r') as fp:
        class_to_idx = json.load(fp)

    return class_names, class_to_idx
