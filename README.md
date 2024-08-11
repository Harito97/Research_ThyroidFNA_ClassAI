# All action in this research can be do with main.py

```bash
watch -n 1 nvidia-smi
# Update real time info of GPU when run command
```

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
# nohup python main.py --config configs/experiment_5.yaml > output_experiment_5.log 2>&1 &
# [1] 346474 # 346474 is the pid of experiment 5
nohup bash -c 'while kill -0 346474 2>/dev/null; do sleep 5; done; nohup python main.py --config configs/experiment_6.yaml > output_experiment_6.log 2>&1 &' > output_run_after_completion.log 2>&1 &
# [2] 354432
# nohup python main.py --config configs/experiment_6.yaml > output_experiment_6.log 2>&1 &
tail -n 50 output_experiment_6.log
```

<!-- ### Ex7:
```bash
nohup python main.py --config configs/experiment_6.yaml > output_experiment_6.log 2>&1 &
tail -n 50 output_experiment_6.log
``` -->

### Train Ex8 Ex9 Ex10 Ex11 Ex12 Ex13
```bash
#!/bin/bash

nohup python main.py --config configs/experiment_8.yaml > output_experiment_8.log 2>&1 && \
nohup python main.py --config configs/experiment_9.yaml > output_experiment_9.log 2>&1 && \
nohup python main.py --config configs/experiment_10.yaml > output_experiment_10.log 2>&1 && \
nohup python main.py --config configs/experiment_11.yaml > output_experiment_11.log 2>&1 && \
nohup python main.py --config configs/experiment_12.yaml > output_experiment_12.log 2>&1 && \
nohup python main.py --config configs/experiment_13.yaml > output_experiment_13.log 2>&1
```

### Train Ex15 Ex16 Ex17 Ex18 Ex19 Ex20
```bash
#!/bin/bash

nohup python main.py --config configs/experiment_15.yaml > output_experiment_15.log 2>&1 && \
nohup python main.py --config configs/experiment_16.yaml > output_experiment_16.log 2>&1 && \
nohup python main.py --config configs/experiment_17.yaml > output_experiment_17.log 2>&1 && \
nohup python main.py --config configs/experiment_18.yaml > output_experiment_18.log 2>&1 && \
nohup python main.py --config configs/experiment_19.yaml > output_experiment_19.log 2>&1 && \
nohup python main.py --config configs/experiment_20.yaml > output_experiment_20.log 2>&1
```
...