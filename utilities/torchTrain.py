import torch
from torch import nn
from tqdm.auto import tqdm
import torchmetrics
from typing import Tuple

class torchTrain:
    def __init__(self) -> None:
        """Initialize a training and testing utility class for PyTorch models."""
        pass

    def train_step(self,
                   model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: nn.Module,
                   accuracy_fn,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device) -> Tuple[float, float]:
        """Perform a single training step.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            data_loader (torch.utils.data.DataLoader): DataLoader for training data.
            loss_fn (nn.Module): The loss function used for training.
            accuracy_fn: A function to calculate accuracy.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (torch.device): The device (CPU or GPU) to perform training on.

        Returns:
            tuple: A tuple containing train loss and train accuracy.
        """

        model.to(device)
        ### Training step
        train_loss, train_acc = 0, 0
        model.train()
        for X, y in data_loader:
            # Send X, y to device
            X, y = X.to(device), y.to(device)

            # Forward pass / Calculating logits
            y_pred = model(X)

            # Loss and accuracy calculation
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

            # Backpropagation and Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Adjust train loss fro number of batches
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)

        return train_loss, train_acc


    def test_step(self,
                  model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module,
                  accuracy_fn,
                  device: torch.device)-> Tuple[float, float]:
        """Perform a single testing step.

        Args:
            model (torch.nn.Module): The PyTorch model to test.
            data_loader (torch.utils.data.DataLoader): DataLoader for testing data.
            loss_fn (nn.Module): The loss function used for testing.
            accuracy_fn: A function to calculate accuracy.
            device (torch.device): The device (CPU or GPU) to perform testing on.

        Returns:
            tuple: A tuple containing test loss and test accuracy.
        """
        ### Testing step
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)

                # Forward pass
                y_pred = model(X)

                # Loss and accuracy calculation
                loss = loss_fn(y_pred, y)
                test_loss += loss
                test_acc += accuracy_fn(y_pred.argmax(dim=1), y)

            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)

        return test_loss, test_acc

    def train(self,
              model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              accuracy_fn: torchmetrics.Accuracy,
              loss_fn: nn.Module = nn.CrossEntropyLoss(),
              epochs: int = 5,
              device = "cpu") -> dict:
        """Train a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            accuracy_fn: A function to calculate accuracy.
            loss_fn (nn.Module, optional): The loss function used for training and testing. Default is nn.CrossEntropyLoss().
            epochs (int, optional): The number of training epochs. Default is 5.
            device (str or torch.device, optional): The device (CPU or GPU) to perform training on. Default is "cpu".

        Returns:
            dict: A dictionary containing training and testing results including losses and accuracies.
        """

        # Create dictionary to store results
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        # Make train and test step
        for epoch in tqdm(range(epochs)):

            train_loss, train_acc = self.train_step(model=model,
                    data_loader=train_dataloader,
                    loss_fn=loss_fn,
                    accuracy_fn=accuracy_fn,
                    optimizer=optimizer,
                    device=device)
            test_loss, test_acc = self.test_step(model=model,
                    data_loader=test_dataloader,
                    loss_fn=loss_fn,
                    accuracy_fn=accuracy_fn,
                    device=device)

            # Store results
            results["train_loss"].append(train_loss.detach().item())
            results["train_acc"].append(train_acc.detach().item())
            results["test_loss"].append(test_loss.detach().item())
            results["test_acc"].append(test_acc.detach().item())

            # Print epoch results
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        return results