# All action in this research can be do with main.py

## Create data versions

Example usage:
```bash
cd /path/to/this/project
nohup python main.py --config configs/data_creator.yaml > output_data_creator.log 2>&1 &
# install nohup if you don't have (or thus use python) 
# change the configs/data_creator.yaml as you want (but must be sustainable)
tail -n 50 output_data_creator.log # to see last 50 lines of output info 
# see the output_data_creator.log more if you want
rm *.log
# remove all log
```
The dataset be create will stay in results/dataset_just_create

## Explore the datasets

Example usage:
```bash
cd /path/to/this/project
nohup python main.py --config configs/data_explore.yaml > output_data_explore.log 2>&1 &
# change the configs/data_explore.yaml as you want (but must be sustainable)
tail -n 50 output_data_explore.log # to see last 50 lines of output info 
# see the output_data_explore.log more if you want
rm *.log
# remove all log
```
Results of data set analysis is in output_dir: /path/to/output/directory of the configs/data_explore.yaml file

## Process the experiments

### Ex1:
```bash
cd /path/to/this/project
nohup python main.py --config configs/experiment_1.yaml > output_experiment_1.log 2>&1 &
# change the configs/experiment_1.yaml as you want (but must be sustainable)
tail -n 50 output_experiment_1.log # to see last 50 lines of output info 
# see the output_experiment_1.log more if you want
rm *.log
# remove all log
```

### Ex2:
```bash
nohup python main.py --config configs/experiment_2.yaml > output_experiment_2.log 2>&1 &
tail -n 50 output_experiment_2.log
```

### Ex3:
```bash
nohup python main.py --config configs/experiment_3.yaml > output_experiment_3.log 2>&1 &
tail -n 50 output_experiment_3.log
```

### Ex4:
```bash
nohup python main.py --config configs/experiment_4.yaml > output_experiment_4.log 2>&1 &
tail -n 50 output_experiment_4.log
```

### Ex5:
```bash
nohup python main.py --config configs/experiment_5.yaml > output_experiment_5.log 2>&1 &
tail -n 50 output_experiment_5.log
```

### Ex6:
```bash
nohup python main.py --config configs/experiment_6.yaml > output_experiment_6.log 2>&1 &
tail -n 50 output_experiment_6.log
```

<!-- ### Ex7:
```bash
nohup python main.py --config configs/experiment_6.yaml > output_experiment_6.log 2>&1 &
tail -n 50 output_experiment_6.log
``` -->
...