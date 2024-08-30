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


class EvaluateClassificationModel:
    def __init__(self, config, model, eval_loader, time_stamp=None):
        if time_stamp is None:
            raise ValueError("time_stamp is required")

        self.model = model
        self.eval_loader = eval_loader

        # Check device availability
        if (
            config["evaluate_para"]["device"] == "cuda"
            and not torch.cuda.is_available()
        ):
            print("CUDA is not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config["evaluate_para"]["device"])

        self.type = config["evaluate_para"]["type"]  # validation or testing
        self.config = config

        # Constructing dynamic paths based on the config
        dir_path = config["info_read"]["dir_path"].format(
            time_stamp=time_stamp,
            model_name=config["evaluate_para"]["model_name"],
            model_type=config["evaluate_para"]["model_type"],
        )
        self.save_path = dir_path
        self.model_best_path = config["evaluate_para"]["model_path"].format(
            dir_path=dir_path, model_path=config["info_read"]["model_path"]
        )

        # Create directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        # Load model weights
        self.model.load_state_dict(
            torch.load(self.model_best_path, map_location=self.device), strict=False
        )

        # Initialize wandb
        wandb.init(
            project=config["wandb"]["project"],
            config=config,
            name=config["wandb"]["name"].format(time_stamp=time_stamp),
            notes=config["wandb"].get("description", ""),
        )

    def evaluate(self):
        all_preds = []
        all_labels = []
        all_probs = []

        self.model.to(self.device)
        self.model.eval()
        with no_grad():
            for inputs, labels in self.eval_loader:
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
        print(f"Classification report:\n{class_report}")
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
                    os.path.join(
                        self.save_path,
                        self.config["info_save"][
                            "evaluate_classification_report"
                        ].format(type=self.config["evaluate_para"]["type"]),
                    )
                ),
                "roc_curve": wandb.Image(
                    os.path.join(
                        self.save_path,
                        self.config["info_save"]["evaluate_roc_curve"].format(
                            type=self.config["evaluate_para"]["type"]
                        ),
                    )
                ),
                "confusion_matrix": wandb.Image(
                    os.path.join(
                        self.save_path,
                        self.config["info_save"]["evaluate_confusion_matrix"].format(
                            type=self.config["evaluate_para"]["type"]
                        ),
                    )
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
        plt.savefig(
            os.path.join(
                self.save_path,
                self.config["info_save"]["evaluate_classification_report"].format(
                    type=self.config["evaluate_para"]["type"]
                ),
            )
        )
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
        plt.savefig(
            os.path.join(
                self.save_path,
                self.config["info_save"]["evaluate_roc_curve"].format(
                    type=self.config["evaluate_para"]["type"]
                ),
            )
        )
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
        plt.savefig(
            os.path.join(
                self.save_path,
                self.config["info_save"]["evaluate_confusion_matrix"].format(
                    type=self.config["evaluate_para"]["type"]
                ),
            )
        )
        plt.close()

    def save_results(self, y_true, y_pred, y_prob):
        # Save evaluation results to CSV
        df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
        df.to_csv(
            os.path.join(
                self.save_path,
                self.config["info_save"]["evaluate_log"].format(
                    type=self.config["evaluate_para"]["type"]
                ),
            ),
            index=False,
        )
