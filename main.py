import argparse
import yaml
from src.experiments import * #experiment_a, experiment_b
from src.data import data_creator

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if config['type'] == 'data_creator':
        data_creator.run(config)
    
    # if config['experiment'] == 'a':
    #     experiment_a.run(config)
    # elif config['experiment'] == 'b':
    #     experiment_b.run(config)
    # Add more experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run medical image classification experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)

    # Eg: python main.py --config configs/experiment_a.yaml
