# src.training.test_model.py
import torch
import os
import textwrap
import time
from src.utils.utils import load_config
from src.utils.helpers import compute_metrics
from src.utils.logger import update_log, save_log_and_visualizations


class TestClassificationModel:
    def __init__(self, config_path=None, config=None, **kwargs):
        self.config = config if config else load_config(config_path, **kwargs)

        if "tester" not in self.config:
            raise ValueError("Config must have `tester` key")

        self.device = torch.device(
            self.config["tester"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.model_type = self.config["tester"]["model_type"]
        self.test_logs_dir = self.config["tester"]["test_logs_dir"]

    def setup(self, model, test_loader, criterion=None):
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.criterion = criterion

    def __prepare_log_dir(self):
        # time_stamp = int(time.time())
        model_path = self.config["tester"]["model_path"]
        time_stamp_when_train_model = (
            os.path.dirname(model_path).split("/")[-1].split("_")[0]
        )
        try:
            time_stamp = int(time_stamp_when_train_model)
        except ValueError:
            print(
                "Cannot convert time stamp to int.\n"
                + "As model path is not in the right format ({time_stamp}_{model_type}/{model_name}.pth)."
            )

        log_dir = os.path.join(
            self.test_logs_dir, f"test_logs_{time_stamp}_{self.model_type}"
        )
        os.makedirs(log_dir, exist_ok=True)
        return log_dir, time_stamp

    # def __save_predictions(self, log_dir, labels, preds):
    #     predictions_path = os.path.join(log_dir, "test_predictions.csv")
    #     save_predictions(predictions_path, labels, preds)

    def test(self):
        log_dir, time_stamp = self.__prepare_log_dir()
        description = self.__create_description(log_dir, time_stamp)
        test_loss, test_f1, test_acc, test_cm, all_labels, all_preds = (
            self.__test_one_epoch()
        )
        # self.__save_predictions(log_dir, all_labels, all_preds)

        # Ghi log và lưu các chỉ số
        logs_info = f"Test Results:\n"
        logs_info += f"  + Test Loss: {test_loss}\n"
        logs_info += f"  + Test F1: {test_f1}\n"
        logs_info += f"  + Test Accuracy: {test_acc}\n"
        logs_info += f"  + Test CM:\n{test_cm}\n"
        logs_info += f"Time stamp: {time_stamp}\n"
        logs_info += f"All labels: {all_labels}\n"
        logs_info += f"All preds: {all_preds}\n"

        update_log(
            os.path.join(log_dir, "test_logs.csv"),
            0,
            test_loss,
            None,
            test_f1,
            None,
            test_acc,
            None,
        )

        with open(os.path.join(log_dir, "description.txt"), "w") as f:
            f.write(description)

        save_log_and_visualizations(log_dir, logs_info, test_cm)

    def __test_one_epoch(self):
        self.model.eval()
        # total_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # total_loss += self.criterion(outputs, labels).item()

                preds = outputs.argmax(dim=1)
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        # test_loss = total_loss / len(self.test_loader)
        test_f1, test_acc, test_cm = compute_metrics(all_labels, all_preds)
        return None, test_f1, test_acc, test_cm, all_labels, all_preds

    def __create_description(self, log_dir, time_stamp):
        return textwrap.dedent(
            f"""
            Model: {self.model_type}
            Device: {self.device}
            Time stamp: {time_stamp}
            Log dir: {log_dir}
        """
        )


#####################################################################

# import torch
# import os
# import textwrap
# import time
# from src.utils.utils import load_config
# from src.utils.helpers import compute_metrics
# from src.utils.logger import update_log, save_log_and_visualizations


# class TestClassificationModel:
#     def __init__(self, config_path=None, config=None, **kwargs):
#         if config is not None:
#             self.config = config
#         else:
#             self.config = load_config(config_path, **kwargs)
#         print("Config loaded")
#         print(self.config)

#         if "tester" not in self.config:
#             raise ValueError("Config must have `tester` key")

#         self.device = torch.device(
#             self.config["tester"]["device"] if torch.cuda.is_available() else "cpu"
#         )
#         self.model_type = self.config["tester"]["model_type"]
#         self.test_logs_dir = self.config["tester"]["test_logs_dir"]
#         print(f"Using device: {self.device}")

#     def help():
#         print("This class is used to test a classification model")
#         print("You can use the following methods:")
#         print("0. initialize with (config_path=None, **kwargs)")
#         print("1. setup(model, test_loader, criterion)")
#         print("2. test() to start testing")

#     def setup(self, model, test_loader, criterion):
#         self.set_model(model)
#         self.set_dataloaders(test_loader)
#         self.set_criterion(criterion)
#         print("Setup completed")

#     def set_dataloaders(self, test_loader):
#         self.test_loader = test_loader
#         print("Set test dataloader")

#     def set_criterion(self, criterion):
#         self.criterion = criterion
#         print(f"Set criterion in {self.device}")

#     def set_model(self, model):
#         self.model = model
#         self.model.to(self.device)
#         print(f"Set model in {self.device}")

#     def __set_log(self):
#         self.time_stamp = int(time.time())
#         self.log_dir = os.path.join(
#             self.test_logs_dir, f"test_logs_{self.time_stamp}_{self.model_type}"
#         )
#         os.makedirs(self.log_dir, exist_ok=True)
#         self.description_log_path = os.path.join(self.log_dir, "description.txt")
#         self.test_logs_txt_path = os.path.join(self.log_dir, "test_logs.txt")
#         self.test_logs_csv_path = os.path.join(self.log_dir, "test_logs.csv")
#         self.test_metrics_path = os.path.join(self.log_dir, "test_metrics.pdf")
#         self.fig_loss_path = os.path.join(self.log_dir, "test_loss_plot.jpg")
#         self.fig_acc_and_f1_path = os.path.join(
#             self.log_dir, "test_acc_and_f1_plot.jpg"
#         )
#         self.test_predictions_path = os.path.join(self.log_dir, "test_predictions.csv")
#         print(f"Set log in {self.log_dir}")

#     def __set_description(self):
#         self.description = textwrap.dedent(
#             f"""
#             Model:                  {self.model_type}
#             Device:                 {self.device}
#             Time stamp:             {self.time_stamp}
#             Log dir:                {self.log_dir}
#             Test logs txt path:     {self.test_logs_txt_path}
#             Test logs csv path:     {self.test_logs_csv_path}
#             Test metrics path:      {self.test_metrics_path}
#             Test predictions path:  {self.test_predictions_path}
#         """
#         )

#     def __log_predictions(self, all_labels, all_preds, all_outputs):
#         # Save predictions and labels to CSV
#         save_predictions(self.test_predictions_path, all_labels, all_preds)
#         print(f"Saved test predictions to {self.test_predictions_path}")

#     def test(self):
#         self.__set_log()
#         self.__set_description()
#         logs_info = ""
#         loss_values = []
#         f1_values = []
#         acc_values = []

#         test_loss, test_f1, test_acc, test_cm, all_labels, all_preds, all_outputs = (
#             self.__test_one_epoch()
#         )

#         # Lưu lại kết quả dự đoán
#         self.__log_predictions(all_labels, all_preds, all_outputs)

#         # Log các chỉ số
#         log_info = (
#             f"Test Results:\n"
#             f"  + Test Loss: {test_loss}\n"
#             f"  + Test F1: {test_f1}\n"
#             f"  + Test Accuracy: {test_acc}\n"
#             f"  + Test CM:\n{test_cm}\n"
#         )
#         logs_info += log_info
#         print(log_info)

#         # Lưu log ra file CSV
#         update_log(
#             self.test_logs_csv_path,
#             0,  # Since this is testing, no epochs, just log as epoch 0
#             test_loss,
#             None,  # No val_loss
#             test_f1,
#             None,  # No val_f1
#             test_acc,
#             None,  # No val_acc
#         )

#         # Gọi hàm lưu description, log và vẽ biểu đồ
#         with open(self.description_log_path, "w") as f:
#             f.write(self.description)
#         save_log_and_visualizations(
#             log_dir=self.log_dir,
#             logs_info=logs_info,
#             train_logs_txt_path=self.test_logs_txt_path,
#             train_logs_csv_path=self.test_logs_csv_path,
#             fig_loss_path=self.fig_loss_path,
#             fig_acc_and_f1_path=self.fig_acc_and_f1_path,
#             cm_dict={"test_cm": test_cm},  # Chỉ có confusion matrix của test
#         )

#     def __test_one_epoch(self):
#         self.model.eval()
#         total_loss = 0
#         all_labels = []
#         all_preds = []
#         all_outputs = []

#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 total_loss += loss.item()

#                 preds = torch.argmax(outputs, dim=1)
#                 all_labels.append(labels.cpu())
#                 all_preds.append(preds.cpu())
#                 all_outputs.append(outputs.cpu())

#         all_labels = torch.cat(all_labels)
#         all_preds = torch.cat(all_preds)
#         all_outputs = torch.cat(all_outputs)
#         test_loss = total_loss / len(self.test_loader)
#         test_f1, test_acc, test_cm = compute_metrics(all_labels, all_preds)
#         return test_loss, test_f1, test_acc, test_cm, all_labels, all_preds, all_outputs
