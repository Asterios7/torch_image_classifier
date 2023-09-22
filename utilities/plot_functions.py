import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import os

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
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
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
