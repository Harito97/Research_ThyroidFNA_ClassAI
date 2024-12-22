import numpy as np

def calculate_metrics(confusion_matrix):
    """
    Tính toán các chỉ số thống kê từ confusion matrix.

    Args:
        confusion_matrix (numpy.ndarray): Ma trận nhầm lẫn (shape: [n_classes, n_classes])

    Returns:
        dict: Tập hợp các chỉ số thống kê cho từng lớp và trung bình.
    """
    # Tổng số lớp
    n_classes = confusion_matrix.shape[0]

    # Khởi tạo các chỉ số
    metrics = {
        "accuracy": None,
        "precision": np.zeros(n_classes),
        "recall": np.zeros(n_classes),
        "f1_score": np.zeros(n_classes),
        "specificity": np.zeros(n_classes),
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1_score": None,
    }

    # Tổng số mẫu
    total_samples = confusion_matrix.sum()
    true_positive = np.diag(confusion_matrix)
    false_positive = confusion_matrix.sum(axis=0) - true_positive
    false_negative = confusion_matrix.sum(axis=1) - true_positive
    true_negative = total_samples - (true_positive + false_positive + false_negative)

    # Accuracy
    metrics["accuracy"] = true_positive.sum() / total_samples

    # Tính Precision, Recall, F1-Score và Specificity cho từng lớp
    for i in range(n_classes):
        tp, fp, fn, tn = true_positive[i], false_positive[i], false_negative[i], true_negative[i]

        # Precision (tp / (tp + fp))
        metrics["precision"][i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall (Sensitivity, tp / (tp + fn))
        metrics["recall"][i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1-score (2 * Precision * Recall / (Precision + Recall))
        if (metrics["precision"][i] + metrics["recall"][i]) > 0:
            metrics["f1_score"][i] = (2 * metrics["precision"][i] * metrics["recall"][i]) / \
                                     (metrics["precision"][i] + metrics["recall"][i])
        else:
            metrics["f1_score"][i] = 0.0

        # Specificity (tn / (tn + fp))
        metrics["specificity"][i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Tính macro-average cho Precision, Recall, F1-Score
    metrics["macro_precision"] = metrics["precision"].mean()
    metrics["macro_recall"] = metrics["recall"].mean()
    metrics["macro_f1_score"] = metrics["f1_score"].mean()

    return metrics

def print_metrics(confusion_matrix):
    # Tính các chỉ số
    metrics = calculate_metrics(confusion_matrix)

    # Hiển thị kết quả
    # print("Confusion Matrix:")
    # print(confusion_matrix)
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

    print('-'*50)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Cua Hieu
    # Tap test truoc 7/2024
    confusion_matrix_Hieu_pre_72024 = np.array([
        [130, 3, 4],  # Class 0
        [4, 77, 28],  # Class 1
        [5, 26, 181]   # Class 2
    ])
    print("Cua Hieu - Tap test truoc 7/2024")
    print_metrics(confusion_matrix_Hieu_pre_72024)

    # Cua Hieu
    # Tap test sau 7/2024
    confusion_matrix_Hieu_after_72024 = np.array([
        [256, 13, 22],  # Class 0
        [30, 96, 189],  # Class 1
        [14, 31, 355]   # Class 2
    ])
    print("Cua Hieu - Tap test sau 7/2024")
    print_metrics(confusion_matrix_Hieu_after_72024)

    # Cua Hai
    # Tap test truoc 7/2024 vs model 428039
    confusion_matrix_Hai_pre_72024_428039 = None
    print("Do luu o cho khac nen hien chua cho vao day")

    # Cua Hai
    # Tap test sau 7/2024 vs model 428039
    confusion_metrics_Hai_after_72024_428039 = np.array([
        [261, 30,  9],
        [ 44, 143, 128],
        [ 16,  92, 292]
    ])
    print("Cua Hai - Tap test sau 7/2024 vs model 428039")
    print_metrics(confusion_metrics_Hai_after_72024_428039)

    # Cua Hai
    # Tap test truoc 7/2024 vs model 492356
    confusion_matrix_Hai_pre_72024_492356 = np.array([
        [57,  2,  2],
        [ 0, 85, 11],
        [ 2,  15, 98]
    ])
    print("Cua Hai - Tap test sau 7/2024 vs model 492356")
    print_metrics(confusion_matrix_Hai_pre_72024_492356)

    # Cua Hai
    # Tap test sau 7/2024 vs model 492356
    confusion_matrix_Hai_after_72024_492356 = np.array([
        [262,  28,  10],
        [ 40, 138, 137],
        [ 17,  83, 300]
    ])
    print("Cua Hai - Tap test sau 7/2024 vs model 492356")
    print_metrics(confusion_matrix_Hai_after_72024_492356)
