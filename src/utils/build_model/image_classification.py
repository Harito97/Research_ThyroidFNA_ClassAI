# src/utils/build_model/image_classification.py

import wandb
import torch


class TrainImageClassificationModel:
    def __init__(
        self,
        experiment_yaml_config,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        patience,
        device,
        save_path,
    ):
        """
        Train an image classification model.

        Args:
            experiment_yaml_config (dict): Configuration for the experiment including logging details.
            model (torch.nn.Module): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            num_epochs (int): Number of epochs to train.
            patience (int): Number of epochs to wait for early stopping.
            device (torch.device): Device to train on.
            save_path (str): Path to save the best model.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        self.save_path = save_path

        # Initialize wandb
        wandb.init(
            project=experiment_yaml_config["logging"]["project_name"],
            config={
                "epochs": num_epochs,
                "learning_rate": optimizer.defaults["lr"],
                "batch_size": train_loader.batch_size,
            },
            name=experiment_yaml_config["logging"]["run_name"],
        )

        # Check save directory exists
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

    def train(self):
        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            corrects = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                corrects += (preds == labels).sum().item()

            train_loss = train_loss / len(self.train_loader.dataset)
            train_acc = corrects / len(self.train_loader.dataset)

            # Log training metrics
            wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"\t+ Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.6f}")

            self.model.eval()
            valid_loss = 0.0
            corrects = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)
                    corrects += (preds == labels).sum().item()

            valid_loss = valid_loss / len(self.val_loader.dataset)
            valid_acc = corrects / len(self.val_loader.dataset)

            # Log validation metrics
            wandb.log({"valid_loss": valid_loss, "valid_accuracy": valid_acc})

            print(
                f"\t+ Validation Loss: {valid_loss:.6f}, Validation Accuracy: {valid_acc:.6f}"
            )

            # Check for early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                # Save model
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Model saved to {self.save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

            self.scheduler.step()

        # Finish wandb run
        wandb.finish()


import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import os
import torch
from torch import no_grad, device


class ValidImageClassificationModel:
    def __init__(self, experiment_yaml_config, model, val_loader):
        self.model = model
        self.val_loader = val_loader
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = experiment_yaml_config["logging"]["save_path"]
        self.model_best_path = experiment_yaml_config["training"]["save_path"]
        self.model.load_state_dict(
            torch.load(self.model_best_path, map_location=self.device), strict=False
        )

        # Create directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Initialize wandb
        wandb.init(
            project=experiment_yaml_config["logging"]["project_name"],
            config={
                # Adjusted for available parameters
                "save_path": self.save_path,
            },
            name=experiment_yaml_config["logging"]["run_name"] + "_validation",
        )

    def evaluate(self):
        all_preds = []
        all_labels = []
        all_probs = []

        self.model.to(self.device)
        self.model.eval()
        with no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Print overall metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Generate classification report
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        self.plot_classification_report(class_report)

        # Generate ROC curve
        self.plot_roc_curve(all_labels, all_probs, num_classes=len(set(all_labels)))

        # Generate and save confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)

        # Save results
        self.save_results(all_labels, all_preds, all_probs)

        # Log final metrics to W&B
        wandb.log(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": wandb.Image(
                    os.path.join(self.save_path, "classification_report.png")
                ),
                "roc_curve": wandb.Image(os.path.join(self.save_path, "roc_curve.png")),
                "confusion_matrix": wandb.Image(
                    os.path.join(self.save_path, "confusion_matrix.png")
                ),
            }
        )
        wandb.finish()

    def plot_classification_report(self, report):
        report_df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            report_df.iloc[:-1, :].astype(float), annot=True, fmt=".2f", cmap="Blues"
        )
        plt.title("Classification Report")
        plt.savefig(os.path.join(self.save_path, "classification_report.png"))
        plt.close()

    def plot_roc_curve(self, y_true, y_prob, num_classes):
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        plt.figure(figsize=(10, 7))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_path, "roc_curve.png"))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.arange(cm.shape[0]),
            yticklabels=np.arange(cm.shape[1]),
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.save_path, "confusion_matrix.png"))
        plt.close()

    def save_results(self, labels, preds, probs):
        # Optionally save detailed results like raw predictions and labels
        results_df = pd.DataFrame(
            {"Label": labels, "Prediction": preds, "Probabilities": list(probs)}
        )
        results_df.to_csv(
            os.path.join(self.save_path, "detailed_results.csv"), index=False
        )
