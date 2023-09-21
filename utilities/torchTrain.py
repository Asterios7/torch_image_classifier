import torch
from torch import nn
from tqdm.auto import tqdm
import torchmetrics


class torchTrain:
    def __init__(self) -> None:
        pass

    def train_step(self,
                   model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: nn.Module,
                   accuracy_fn,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device):
        
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
                  device: torch.device):
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
              device = "cpu"):

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