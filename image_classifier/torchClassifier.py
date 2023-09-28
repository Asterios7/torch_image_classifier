import torch
from torch import nn
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List
import mlflow
from PIL import Image


class torchClassifier:
    def __init__(self) -> None:
        """
        Initialize a training and testing utility class for PyTorch models.
        """
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
            data_loader (torch.utils.data.DataLoader): DataLoader for
                training data.
            loss_fn (nn.Module): The loss function used for training.
            accuracy_fn: A function to calculate accuracy.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (torch.device): The device (CPU or GPU) to perform
                training on.

        Returns:
            tuple: A tuple containing train loss and train accuracy.
        """

        model.to(device)
        # Training step
        train_loss, train_acc = 0, 0
        model.train()
        for X, y in data_loader:
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
                  device: torch.device
                  ) -> Tuple[float, float]:
        """Perform a single testing step.

        Args:
            model (torch.nn.Module): The PyTorch model to test.
            data_loader (torch.utils.data.DataLoader): DataLoader for
                testing data.
            loss_fn (nn.Module): The loss function used for testing.
            accuracy_fn: A function to calculate accuracy.
            device (torch.device): The device (CPU or GPU) to perform
                testing on.

        Returns:
            tuple: A tuple containing test loss and test accuracy.
        """
        # Testing step
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

    def fit(self,
            model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            accuracy_fn: torchmetrics.Accuracy,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            epochs: int = 5,
            device: torch.device = "cpu",
            mlflow_experiment: str = "torch_classifier",
            mlfow_run_name: str = "wicked_jonathan") -> dict:
        """Train a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for
            training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for
            testing data.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            accuracy_fn: A function to calculate accuracy.
            loss_fn (nn.Module, optional): The loss function used for training
            and testing. Default is nn.CrossEntropyLoss().
            epochs (int, optional): The number of training epochs. Default=5.
            device (str or torch.device, optional): The device (CPU or GPU) to
            perform training on. Default is "cpu".

        Returns:
            dict: A dictionary containing training and testing results
            including losses and accuracies.
        """
        mlflow.set_experiment(mlflow_experiment)
        experiment = mlflow.get_experiment_by_name(mlflow_experiment)
        # Start an MLflow run
        with mlflow.start_run(experiment_id=experiment.experiment_id,
                              run_name=mlfow_run_name):

            # Create dictionary to store results
            results = {"train_loss": [],
                       "train_acc": [],
                       "test_loss": [],
                       "test_acc": []}

            # Make train and test step
            for epoch in range(epochs):

                train_loss, train_acc = self.train_step(
                        model=model,
                        data_loader=train_dataloader,
                        loss_fn=loss_fn,
                        accuracy_fn=accuracy_fn,
                        optimizer=optimizer,
                        device=device
                        )
                test_loss, test_acc = self.test_step(
                        model=model,
                        data_loader=test_dataloader,
                        loss_fn=loss_fn,
                        accuracy_fn=accuracy_fn,
                        device=device
                        )

                # Store results
                results["train_loss"].append(train_loss.detach().item())
                results["train_acc"].append(train_acc.detach().item())
                results["test_loss"].append(test_loss.detach().item())
                results["test_acc"].append(test_acc.detach().item())

                # Print epoch results
                print(
                    f"Epoch: {epoch} |"
                    f"Train loss: {train_loss:.4f} | "
                    f"Train acc: {train_acc:.4f} | "
                    f"Test loss: {test_loss:.4f} | "
                    f"Test acc: {test_acc:.4f}"
                    )

                # Log metrics and params to MLflow
                mlflow.log_metric("train_loss", train_loss.detach().item(),
                                  step=epoch)
                mlflow.log_metric("train_acc", train_acc.detach().item(),
                                  step=epoch)
                mlflow.log_metric("test_loss", test_loss.detach().item(),
                                  step=epoch)
                mlflow.log_metric("test_acc", test_acc.detach().item(),
                                  step=epoch)
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("optimizer", optimizer.__class__.__name__)
            mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
            mlflow.end_run()

        return results

    def predict(self,
                model: torch.nn.Module,
                image_path: str,
                class_names: List[str],
                image_size: Tuple[int, int] = (224, 224),
                transform: torchvision.transforms = None,
                device: torch.device = 'cpu'):
        """
        Predict the class label and probability of an input image
        using a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model for image
            classification.
            image_path (str): The file path to the input image.
            class_names (List[str]): List of class names, ordered by the
            model's output.
            image_size (Tuple[int, int], optional): The target image size for
            resizing (width, height).
                Defaults to (224, 224).
            transform (torchvision.transforms, optional): A custom image
            transformation pipeline.
                If not provided, a default transformation will be used.
            device (torch.device, optional): The device (CPU or GPU) to
            perform inference on.
                Defaults to 'cpu'.

        Returns:
            Tuple[str, float]: A tuple containing the predicted class label
            and its corresponding probability.

        Example:
            model = YourImageClassificationModel()
            image_path = 'path/to/your/image.jpg'
            class_names = ['class1', 'class2', ...]
            predicted_class, predicted_probability = model.predict(model,
            image_path, class_names)
            print(f'Predicted Class: {predicted_class}')
            print(f'Predicted Probability: {predicted_probability}')

        """

        # Open the image with PIL
        img = Image.open(image_path)
        # Create a transform if None is provided
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])

        model.to(device)

        # Predict
        model.eval()
        with torch.inference_mode():
            # Transform data and adding batch dimension
            transformed_image = image_transform(img).unsqueeze(dim=0)
            # Make prediction
            image_pred = model(transformed_image.to(device))
        # Logits to probs
        image_pred_probs = torch.softmax(image_pred, dim=1)

        # 9. Convert the model's pred probs to pred labels
        image_pred_label_idx = torch.argmax(image_pred_probs,
                                            dim=1)
        pred_class = class_names[image_pred_label_idx]
        pred_probability = round(image_pred_probs.max().cpu().item(), 3)

        return pred_class, pred_probability
