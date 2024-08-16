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
from PIL import Image


class Validator:
    def __init__(self, experiment_yaml_config, model, data_dir: str):
        """Data dir has B2/ B5/ B6/"""
        self.model = model
        self.data_dir = data_dir
        self.x = []
        self.y = []
        self.batch_size = experiment_yaml_config["data"]["batch_size"]
        self.load_x_y()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = experiment_yaml_config["logging"]["save_path"]

        # Create directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Initialize wandb
        wandb.init(
            project=experiment_yaml_config["logging"]["project_name"],
            config={
                "save_path": self.save_path,
            },
            name=experiment_yaml_config["logging"]["run_name"] + "_validation",
        )

    def load_x_y(self):
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            label = self.label_to_index(class_dir)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                self.x.append(img_path)
                self.y.append(label)

    def evaluate(self):
        all_preds = []
        all_labels = []
        all_probs = []
        top2_correct = 0
        total_samples = 0

        print(
            f"About the dataset:\nNumber of samples: {len(self.x)}\nNumber of classes: {len(set(self.y))}"
        )

        self.model.to(self.device)
        self.model.eval()
        self.model.eval_mode()
        with torch.no_grad():
            for i in range(0, len(self.x), self.batch_size):
                batch_x = self.x[i : min(i + self.batch_size, len(self.x))]
                batch_y = self.y[i : min(i + self.batch_size, len(self.y))]
                outputs = self.model.predict_from_path(batch_x)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y)
                all_probs.extend(outputs.cpu().numpy())
                print(
                    f"Batch {i // self.batch_size + 1} / {len(self.x) // self.batch_size}:"
                )
                print(
                    f"Precisions: {preds.cpu().numpy()}\nLabels: {batch_y}" #\nPobabilities: {outputs.cpu().numpy()}"
                )

                for label, output in zip(batch_y, outputs):
                    top2_preds = torch.topk(output, 2)[
                        1
                    ]  # No need for dim when working with 1D tensors
                    if label in top2_preds.cpu().numpy():
                        top2_correct += 1
                    total_samples += 1

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)  # self.y)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        top2_accuracy = top2_correct / total_samples

        # Print overall metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Top-2 Accuracy: {top2_accuracy:.4f}")

        # Generate classification report
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        self.plot_classification_report(class_report)

        # Generate ROC curve
        self.plot_roc_curve(all_labels, all_probs, num_classes=len(set(all_labels)))

        # Generate and save confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)

        # Save results
        self.save_results(all_labels, all_preds, all_probs)

        # Plot per-class metrics
        self.plot_per_class_metrics(all_labels, all_preds, class_report, top2_accuracy)

        # Log final metrics to W&B
        wandb.log(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "top2_accuracy": top2_accuracy,
                "classification_report": wandb.Image(
                    os.path.join(self.save_path, "classification_report.png")
                ),
                "roc_curve": wandb.Image(os.path.join(self.save_path, "roc_curve.png")),
                "confusion_matrix": wandb.Image(
                    os.path.join(self.save_path, "confusion_matrix.png")
                ),
                "f1_score_per_class": wandb.Image(
                    os.path.join(self.save_path, "f1_score_per_class.png")
                ),
                "accuracy_per_class": wandb.Image(
                    os.path.join(self.save_path, "accuracy_per_class.png")
                ),
                "top2_accuracy_per_class": wandb.Image(
                    os.path.join(self.save_path, "top2_accuracy_per_class.png")
                ),
                "metrics_all_data": wandb.Image(
                    os.path.join(self.save_path, "metrics_all_data.png")
                ),
            }
        )
        wandb.finish()

    def label_to_index(self, label):
        """Convert label string to numeric index"""
        label_map = {"B2": 0, "B5": 1, "B6": 2}  # Adjust as necessary
        return label_map.get(label, -1)

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

    def plot_per_class_metrics(
        self, all_labels, all_preds, class_report, top2_accuracy
    ):
        metrics = {}
        for label, metrics_dict in class_report.items():
            if label == "accuracy" or label == "macro avg" or label == "weighted avg":
                continue
            metrics[label] = {
                "F1 Score": metrics_dict.get("f1-score", 0),
                "Accuracy": metrics_dict.get("support", 0)
                / sum(
                    [
                        v.get("support", 0)
                        for k, v in class_report.items()
                        if k not in ["accuracy", "macro avg", "weighted avg"]
                    ]
                ),
            }

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.sort_index(inplace=True)

        plt.figure(figsize=(10, 7))
        sns.barplot(x=metrics_df.index, y="F1 Score", data=metrics_df)
        plt.title("F1 Score per Class")
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.save_path, "f1_score_per_class.png"))
        plt.close()

        plt.figure(figsize=(10, 7))
        sns.barplot(x=metrics_df.index, y="Accuracy", data=metrics_df)
        plt.title("Accuracy per Class")
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.save_path, "accuracy_per_class.png"))
        plt.close()

        plt.figure(figsize=(10, 7))
        top2_acc_per_class = {label: top2_accuracy for label in metrics_df.index}
        top2_acc_df = pd.DataFrame(
            top2_acc_per_class.items(), columns=["Class", "Top-2 Accuracy"]
        )
        sns.barplot(x="Class", y="Top-2 Accuracy", data=top2_acc_df)
        plt.title("Top-2 Accuracy per Class")
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.save_path, "top2_accuracy_per_class.png"))
        plt.close()

        all_data_metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Top-2 Accuracy",
                ],
                "Value": [
                    accuracy_score(all_labels, all_preds),
                    precision_score(all_labels, all_preds, average="weighted"),
                    recall_score(all_labels, all_preds, average="weighted"),
                    f1_score(all_labels, all_preds, average="weighted"),
                    top2_accuracy,
                ],
            }
        )

        plt.figure(figsize=(10, 7))
        sns.barplot(x="Metric", y="Value", data=all_data_metrics_df)
        plt.title("Metrics for All Data")
        plt.savefig(os.path.join(self.save_path, "metrics_all_data.png"))
        plt.close()

    def save_results(self, labels, preds, probs):
        results_df = pd.DataFrame(
            {"Label": labels, "Prediction": preds, "Probabilities": list(probs)}
        )
        results_df.to_csv(
            os.path.join(self.save_path, "detailed_results.csv"), index=False
        )
