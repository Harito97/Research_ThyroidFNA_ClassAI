#!/bin/bash
# in ./ project directory

# Chạy từng lệnh tuần tự với nohup để tiếp tục chạy ngay cả khi mất kết nối
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_vgg16_data_701515.yaml > logs/no_train_aug_no_val_aug_vgg16_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_vgg19_data_701515.yaml > logs/no_train_aug_no_val_aug_vgg19_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_resnet18_data_701515.yaml > logs/no_train_aug_no_val_aug_resnet18_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_resnet152_data_701515.yaml > logs/no_train_aug_no_val_aug_resnet152_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_densenet121_data_701515.yaml > logs/no_train_aug_no_val_aug_densenet121_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_densenet201_data_701515.yaml > logs/no_train_aug_no_val_aug_densenet201_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_efficientnet_b0_data_701515.yaml > logs/no_train_aug_no_val_aug_efficientnet_b0_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_efficientnet_b7_data_701515.yaml > logs/no_train_aug_no_val_aug_efficientnet_b7_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_mobilenet_v1_data_701515.yaml > logs/no_train_aug_no_val_aug_mobilenet_v1_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_mobilenet_v3_large_data_701515.yaml > logs/no_train_aug_no_val_aug_mobilenet_v3_large_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_vit_b_16_data_701515.yaml > logs/no_train_aug_no_val_aug_vit_b_16_data_701515.log 2>&1
nohup python experiments/train_module1 --config_path configs/no_train_aug_no_val_aug/train_module1_vit_l_16_data_701515.yaml > logs/no_train_aug_no_val_aug_vit_l_16_data_701515.log 2>&1