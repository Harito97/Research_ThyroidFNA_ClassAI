import wandb
import torch
import os
import csv
import matplotlib.pyplot as plt
import time


class TrainClassificationModel:
    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
    ):
        """
        Train an image classification model.

        Args:
            config (dict): Configuration for the experiment including logging details.
            model (torch.nn.Module): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = config["training"]["num_epochs"]
        self.patience = config["training"]["patience"]
        self.device = torch.device(
            config["training"]["device"] if torch.cuda.is_available() else "cpu"
        )

        # Generate time_stamp and update config
        time_stamp = int(time.time())
        config["time_stamp"] = time_stamp

        # Update paths with time_stamp and model name/type
        config["info_save"]["dir_path"] = config["info_save"]["dir_path"].format(
            time_stamp=config["time_stamp"],
            model_name=config["training"]["model_name"],
            model_type=config["training"]["model_type"],
        )
        self.save_path = os.path.join(
            config["info_save"]["dir_path"], config["info_save"]["model_path"]
        )
        self.csv_log_path = os.path.join(
            config["info_save"]["dir_path"], config["info_save"]["training_log"]
        )
        self.metrics_path = os.path.join(
            config["info_save"]["dir_path"], config["info_save"]["training_metrics"]
        )

        # Initialize wandb
        wandb.init(
            project=config["wandb"]["project"],
            config={
                "epochs": self.num_epochs,
                "learning_rate": optimizer.defaults["lr"],
                "batch_size": train_loader.batch_size,
            },
            name=config["wandb"]["name"].format(time_stamp=config["time_stamp"]),
            notes=config["wandb"].get("description", ""),
        )

        # Check if save directory exists, if not, create it
        os.makedirs(config["info_save"]["dir_path"], exist_ok=True)

        # Initialize CSV logging
        self.init_csv_log()

    def init_csv_log(self):
        with open(self.csv_log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "Train Loss",
                    "Train Accuracy",
                    "Validation Loss",
                    "Validation Accuracy",
                    "Learning Rate",
                ]
            )

    def update_csv_log(self, epoch, train_loss, train_acc, valid_loss, valid_acc, lr):
        with open(self.csv_log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc, lr])

    def plot_metrics(self, train_losses, valid_losses, train_accs, valid_accs):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, "bo-", label="Train Loss")
        plt.plot(epochs, valid_losses, "ro-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, "bo-", label="Train Accuracy")
        plt.plot(epochs, valid_accs, "ro-", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        # Save the figure to wandb
        wandb.log({"Training Metrics": wandb.Image(plt)})
        plt.savefig(self.metrics_path)
        plt.close()

    def train(self):
        best_valid_loss = float("inf")
        patience_counter = 0

        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []

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
            train_losses.append(train_loss)
            train_accs.append(train_acc)

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
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            # Log training metrics & validation metrics
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                }
            )

            print(
                f"\t+ Validation Loss: {valid_loss:.6f}, Validation Accuracy: {valid_acc:.6f}"
            )

            # Update CSV log
            self.update_csv_log(
                epoch + 1,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                self.optimizer.param_groups[0]["lr"],
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

        # Plot and save metrics
        self.plot_metrics(train_losses, valid_losses, train_accs, valid_accs)

        # Finish wandb run
        wandb.finish()
