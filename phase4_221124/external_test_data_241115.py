import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Hàm chính
def evaluate_model(pth_path, folder_path, output_dir="output_results"):
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục lưu kết quả nếu chưa có

    # 1. Load model
    num_classes = len(os.listdir(folder_path))  # Số nhãn từ folder
    model = torch.hub.load('pytorch/vision', 'efficientnet_b0', weights=None) # pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    # 2. Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=12 * 10, shuffle=False)

    # 3. Dự đoán
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.append(probs.numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Save all_preds and all_labels
    np.save(os.path.join(output_dir, "all_preds.npy"), all_preds)
    np.save(os.path.join(output_dir, "all_labels.npy"), all_labels)


    # 4. Tính chỉ số
    y_true = np.eye(num_classes)[all_labels]  # One-hot encoding
    aucs = [roc_auc_score(y_true[:, i], all_preds[:, i]) for i in range(num_classes)]
    fpr, tpr, _ = roc_curve(y_true.ravel(), all_preds.ravel())
    accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')
    report = classification_report(all_labels, np.argmax(all_preds, axis=1), target_names=dataset.classes)
    cm = confusion_matrix(all_labels, np.argmax(all_preds, axis=1))

    # Ghi kết quả vào file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUCs: {aucs}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # 5. Vẽ biểu đồ
    # (1) ROC Curve
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], all_preds[:, i])
        plt.plot(fpr, tpr, label=f"Class {dataset.classes[i]} (AUC = {aucs[i]:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))

    # (2) Confusion Matrix
    plt.figure(figsize=(8, 8))  # Tăng kích thước để hiển thị rõ ràng hơn
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(dataset.classes))
    plt.xticks(tick_marks, dataset.classes, rotation=0)
    plt.yticks(tick_marks, dataset.classes)

    # Thêm số liệu lên từng ô
    thresh = cm.max() / 2.0  # Ngưỡng để chọn màu chữ phù hợp (đen hoặc trắng)
    for i in range(len(dataset.classes)):
        for j in range(len(dataset.classes)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))


# Chạy hàm
# model_path = "/Data/Projects/TempDataLogs_FromServer/1726428039_efficientnet_b0/best_f1_model.pth"
# data_path = "/Data/Projects/TempDataLogs_FromServer/data/data_test_241115"
# evaluate_model(model_path, data_path, output_dir="output_results_1726428039_efficientnet_b0")

model_path = "/Data/Projects/TempDataLogs_FromServer/1726492356_efficientnet_b0/best_f1_model.pth"
data_path = "/Data/Projects/TempDataLogs_FromServer/data/data_test_241115"
evaluate_model(model_path, data_path, output_dir="output_results_1726492356_efficientnet_b0")
