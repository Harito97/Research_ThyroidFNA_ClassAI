# src.training.train_model.py content
import torch
import textwrap
import os
import time
import torch.nn.functional as F
from src.utils.utils import load_config
from src.utils.helpers import compute_metrics
from src.utils.logger import update_log, save_log_and_visualizations, save_model


class TrainClassificationModel:
    def __init__(self, config_path=None, config=None,**kwargs):
        if config is not None:
            config = config
        else:
            config = load_config(config_path, **kwargs)
        print("Config loaded")
        print(config)
        if "trainer" not in config:
            raise ValueError("Config must have `trainer` key")
        self.model_type = config["trainer"]["model_type"]
        self.num_epochs = config["trainer"]["num_epochs"]
        self.patience = config["trainer"]["patience"]
        self.device = torch.device(
            config["trainer"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.best_loss = float("inf")
        self.best_f1 = 0
        self.best_acc = 0
        self.patience_counter = 0
        self.best_loss_cm = None
        self.best_f1_cm = None
        self.best_acc_cm = None

    def help():
        print("This class is used to train a classification model")
        print("You can use the following methods:")
        print("0. initialize with (config_path=None, **kwargs)")
        print(
            "1. setup(model, train_loader, val_loader, criterion, optimizer, scheduler)"
        )
        print("2. reset() if needed")
        print("3. train() to start training")

    def setup(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        self.set_model(model)
        self.set_dataloaders(train_loader, val_loader)
        self.set_criterion(criterion)
        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)
        print("Setup completed")

    def reset(self):
        self.best_loss = float("inf")
        self.best_f1 = 0
        self.best_acc = 0
        self.patience_counter = 0

    def set_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        print("Set dataloaders")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        print(f"Set optimizer in {self.device}")

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        print(f"Set scheduler in cpu")

    def set_criterion(self, criterion):
        self.criterion = criterion
        print(f"Set criterion in cpu")

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        print(f"Set model in {self.device}")

    def __set_log(self):
        self.time_stamp = int(time.time())
        self.log_dir = f"logs/{self.time_stamp}_{self.model_type}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.description_log_path = os.path.join(self.log_dir, "description.txt")
        self.train_logs_txt_path = os.path.join(self.log_dir, "train_logs.txt")
        self.train_logs_csv_path = os.path.join(self.log_dir, "train_logs.csv")
        self.train_metrics_path = os.path.join(self.log_dir, "train_metrics.pdf")
        self.fig_loss_path = os.path.join(self.log_dir, "loss_plot.jpg")
        self.fig_acc_and_f1_path = os.path.join(self.log_dir, "acc_and_f1_plot.jpg")
        self.fig_best_loss_cm_path = os.path.join(self.log_dir, "best_loss_cm.jpg")
        self.fig_best_f1_cm_path = os.path.join(self.log_dir, "best_f1_cm.jpg")
        self.fig_best_acc_cm_path = os.path.join(self.log_dir, "best_acc_cm.jpg")
        self.best_loss_model_path = os.path.join(self.log_dir, "best_loss_model.pth")
        self.best_f1_model_path = os.path.join(self.log_dir, "best_f1_model.pth")
        self.best_acc_model_path = os.path.join(self.log_dir, "best_acc_model.pth")
        print(f"Set log in {self.log_dir}")

    def __set_description(self):
        self.description = textwrap.dedent(
            f"""
            Model:                  {self.model_type}
            Num epochs:             {self.num_epochs}
            Patience:               {self.patience}
            Device:                 {self.device}
            Time stamp:             {self.time_stamp}
            Log dir:                {self.log_dir}
            Train logs txt path:    {self.train_logs_txt_path}
            Train logs csv path:    {self.train_logs_csv_path}
            Train metrics path:     {self.train_metrics_path}
            Fig loss path:          {self.fig_loss_path}
            Fig acc and f1 path:    {self.fig_acc_and_f1_path}
            Fig best loss cm path:  {self.fig_best_loss_cm_path}
            Fig best f1 cm path:    {self.fig_best_f1_cm_path}
            Fig best acc cm path:   {self.fig_best_acc_cm_path}
            Best loss model path:   {self.best_loss_model_path}
            Best f1 model path:     {self.best_f1_model_path}
            Best acc model path:    {self.best_acc_model_path}
        """
        )

    def __early_stopping(self, val_loss, val_f1, val_acc, val_cm, epoch):
        # Kiểm tra điều kiện lưu model
        flag = [False] * 3
        if val_loss < self.best_loss:
            flag[0] = True
            self.best_loss = val_loss
            save_model(self.model, self.best_loss_model_path)
            self.best_loss_cm = val_cm
            print(f"**Update best model** about loss in epoch: {epoch+1}")
        if val_f1 > self.best_f1:
            flag[1] = True
            self.best_f1 = val_f1
            save_model(self.model, self.best_f1_model_path)
            self.best_f1_cm = val_cm
            print(f"**Update best model** about f1 in epoch: {epoch+1}")
        if val_acc > self.best_acc:
            flag[2] = True
            self.best_acc = val_acc
            save_model(self.model, self.best_acc_model_path)
            self.best_acc_cm = val_cm
            print(f"**Update best model** about acc in epoch: {epoch+1}")

        # Kiểm tra dừng sớm
        if not any(flag):
            self.patience_counter += 1
        else:
            self.patience_counter = 0

    def train(self):
        self.__set_log()
        self.__set_description()
        logs_info = ""
        loss_values = []
        f1_values = []
        acc_values = []
        cm_dict = {}

        for epoch in range(self.num_epochs):
            train_loss, train_f1, train_acc, train_cm = self.__train_one_epoch()
            val_loss, val_f1, val_acc, val_cm = self.__validate()

            # Lưu lại giá trị để vẽ biểu đồ sau cùng
            loss_values.append((train_loss, val_loss))
            f1_values.append((train_f1, val_f1))
            acc_values.append((train_acc, val_acc))

            # Log các chỉ số
            log_info = (
                f"Epoch {epoch + 1}:\n"
                f"  + Train Loss: {train_loss}\n"
                f"  + Val Loss: {val_loss}\n"
                f"  + Train F1: {train_f1}\n"
                f"  + Val F1: {val_f1}\n"
                f"  + Train Accuracy: {train_acc}\n"
                f"  + Val Accuracy: {val_acc}\n"
                f"  + Train CM:\n{train_cm}\n"
                f"  + Val CM:\n{val_cm}"
            )
            logs_info += log_info
            print(log_info)

            update_log(
                self.train_logs_csv_path,
                epoch,
                train_loss,
                val_loss,
                train_f1,
                val_f1,
                train_acc,
                val_acc,
            )

            # Early stopping
            self.__early_stopping(val_loss, val_f1, val_acc, val_cm, epoch)
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Tạo dictionary lưu confusion matrix cho best loss, best f1, và best acc
        cm_dict["best_loss"] = self.best_loss_cm
        cm_dict["best_f1"] = self.best_f1_cm
        cm_dict["best_acc"] = self.best_acc_cm

        # Gọi hàm lưu description, log và vẽ biểu đồ
        with open(self.description_log_path, "w") as f:
            f.write(self.description)
        save_log_and_visualizations(
            log_dir=self.log_dir,
            logs_info=logs_info,
            train_logs_txt_path=self.train_logs_txt_path,
            train_logs_csv_path=self.train_logs_csv_path,
            fig_loss_path=self.fig_loss_path,
            fig_acc_and_f1_path=self.fig_acc_and_f1_path,
            fig_best_loss_cm_path=self.fig_best_loss_cm_path,
            fig_best_f1_cm_path=self.fig_best_f1_cm_path,
            fig_best_acc_cm_path=self.fig_best_acc_cm_path,
            cm_dict=cm_dict,
        )

    def __train_one_epoch(self):
        self.model.train()
        total_loss = 0
        all_labels = []
        all_preds = []

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        train_loss = total_loss / len(self.train_loader)
        train_f1, train_acc, train_cm = compute_metrics(all_labels, all_preds)
        return train_loss, train_f1, train_acc, train_cm

    def __validate(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        val_loss = total_loss / len(self.val_loader)
        val_f1, val_acc, val_cm = compute_metrics(all_labels, all_preds)
        return val_loss, val_f1, val_acc, val_cm
