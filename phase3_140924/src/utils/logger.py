import os
import csv
import pandas as pd
import torch
from fpdf import FPDF
from src.visualization.visualize import plot_loss, plot_acc_and_f1
from src.visualization.plot_metrics import plot_confusion_matrix


def update_log(
    log_csv_path, epoch, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc
):
    """
    Lưu log mỗi epoch vào file CSV.
    """
    file_exists = os.path.isfile(log_csv_path)

    with open(log_csv_path, mode="a", newline="") as log_file:
        writer = csv.writer(log_file)
        if not file_exists:
            writer.writerow(
                [
                    "Epoch",
                    "Train Loss",
                    "Val Loss",
                    "Train F1",
                    "Val F1",
                    "Train Accuracy",
                    "Val Accuracy",
                ]
            )
        writer.writerow(
            [epoch + 1, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc]
        )


def save_log_and_visualizations(
    log_dir,
    logs_info,
    train_logs_txt_path,
    train_logs_csv_path,
    fig_loss_path,
    fig_acc_and_f1_path,
    fig_best_loss_cm_path,
    fig_best_f1_cm_path,
    fig_best_acc_cm_path,
    cm_dict,
):
    """
    Lưu log ra file txt và tổng hợp hình ảnh về loss, f1, accuracy và confusion matrix.
    """
    # Lưu log dạng text
    with open(train_logs_txt_path, "x") as f:
        f.write(logs_info)

    # Tạo DataFrame từ log CSV để dùng cho việc vẽ
    df_logs = pd.read_csv(train_logs_csv_path)

    # Vẽ loss
    plot_loss(df_logs, fig_loss_path)

    # Vẽ F1 và Accuracy
    plot_acc_and_f1(df_logs, fig_acc_and_f1_path)

    # Lưu confusion matrix cho best loss, f1, và accuracy
    if "best_loss" in cm_dict:
        plot_confusion_matrix(
            cm_dict["best_loss"],
            title="Confusion Matrix (Best Loss)",
            save_path=fig_best_loss_cm_path,
        )
    if "best_f1" in cm_dict:
        plot_confusion_matrix(
            cm_dict["best_f1"],
            title="Confusion Matrix (Best F1)",
            save_path=fig_best_f1_cm_path,
        )
    if "best_acc" in cm_dict:
        plot_confusion_matrix(
            cm_dict["best_acc"],
            title="Confusion Matrix (Best Accuracy)",
            save_path=fig_best_acc_cm_path,
        )

    # Tổng hợp tất cả hình vào PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Thêm tiêu đề cho các biểu đồ
    pdf.cell(200, 10, txt="Training Metrics", ln=True, align="C")

    # Thêm loss plot
    pdf.image(fig_loss_path, x=10, y=30, w=180)

    # Thêm f1 và accuracy plot
    pdf.add_page()
    pdf.image(fig_acc_and_f1_path, x=10, y=30, w=180)

    # Thêm confusion matrices
    pdf.add_page()
    pdf.image(fig_best_loss_cm_path, x=10, y=30, w=180)
    pdf.add_page()
    pdf.image(fig_best_f1_cm_path, x=10, y=30, w=180)
    pdf.add_page()
    pdf.image(fig_best_acc_cm_path, x=10, y=30, w=180)

    # Lưu PDF
    pdf_path = os.path.join(log_dir, "train_metrics.pdf")
    pdf.output(pdf_path)
    print(f"Saved log and visualizations at: {pdf_path}")


def save_model(model, filename):
    """
    Lưu mô hình vào file.

    Args:
        model (torch.nn.Module): Mô hình PyTorch cần lưu.
        filename (str): Tên file để lưu mô hình.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
