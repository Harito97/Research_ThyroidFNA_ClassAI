For short: 87% in test set (after pass module 1 + module 2).

<!-- **All action in this research can be do with main.py**
```bash
cd /path/to/this/project/phase2
```

# See detail about hardware when train
```bash
watch -n 1 nvidia-smi
# Update real time info of GPU when run command
htop
```

# Create data versions
```bash
nohup python main.py --config configs/data_creator.yaml > logs/output_data_creator.log 2>&1 &
```
The dataset be create will stay in data/dataset_just_create

# Explore the datasets

## Before passing module 1
```bash
nohup python main.py --config configs/data_explorer_before_module1.yaml > logs/output_data_explorer_before_module1.log 2>&1 &
```
Results of data set analysis is in output_dir: /path/to/output/directory of the configs/data_explorer_before_module1.yaml file

## Before passing module 2
With the same is:
```bash
nohup python main.py --config configs/data_explorer_before_module2.yaml > logs/output_data_explorer_before_module2.log 2>&1 &
```

## After passing model (module 1 + module 2)
If with model Heffann3997:
```bash
nohup python main.py --config configs/data_explorer_after_heffann3997.yaml > logs/output_data_explorer_after_heffann3997.log 2>&1 &
```
If with model Hefftrans1363:
```bash
nohup python main.py --config configs/data_explorer_after_hefftrans1363.yaml > logs/output_data_explorer_after_hefftrans1363.log 2>&1 &
```

# Make experiments

These will train the module 1 of the model

## One experiment
```bash
nohup python main.py --config configs/experiment_{x}.yaml > logs/output_experiment_{x}.log 2>&1 &
```
with x is id of experiment - see in configs file.

## Multiple experiment
```bash
nohup python main.py --config configs/experiment_{id1}.yaml > logs/output_experiment_{id1}.log 2>&1 && \
nohup python main.py --config configs/experiment_{id2}.yaml > logs/output_experiment_{id2}.log 2>&1
``` -->
