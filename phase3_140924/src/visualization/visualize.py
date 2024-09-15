# visualize.py content
import matplotlib.pyplot as plt


def plot_loss(df_logs, fig_loss_path):
    """
    Vẽ biểu đồ Loss theo thời gian (epoch) và lưu vào file.
    """
    plt.figure()
    plt.plot(df_logs["Epoch"], df_logs["Train Loss"], label="Train Loss")
    plt.plot(df_logs["Epoch"], df_logs["Val Loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fig_loss_path)
    plt.close()


def plot_acc_and_f1(df_logs, fig_acc_and_f1_path):
    """
    Vẽ biểu đồ F1 score và Accuracy theo thời gian (epoch) và lưu vào file.
    """
    plt.figure()
    plt.plot(df_logs["Epoch"], df_logs["Train F1"], label="Train F1")
    plt.plot(df_logs["Epoch"], df_logs["Val F1"], label="Val F1")
    plt.plot(df_logs["Epoch"], df_logs["Train Accuracy"], label="Train Accuracy")
    plt.plot(df_logs["Epoch"], df_logs["Val Accuracy"], label="Val Accuracy")
    plt.title("F1 Score and Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(fig_acc_and_f1_path)
    plt.close()
