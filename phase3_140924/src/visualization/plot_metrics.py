# plot_metrics.py content
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, title, save_path):
    """
    Vẽ Confusion Matrix và lưu vào file.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(save_path)
    plt.close()
