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

...