import plotly.graph_objects as go
from plotly.subplots import make_subplots

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