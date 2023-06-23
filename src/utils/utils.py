import os
import yaml
import torch
import numpy
import random
from pprint import pprint
from dotmap import DotMap


def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

def load_config(config_name):
    # check that yaml file exists
    if not os.path.exists(config_name):
        raise NameError("YAML configuration file does not exist, exiting!")

    # load the config yaml
    with open(config_name) as f:
        config_yaml = yaml.safe_load(f)

    config = DotMap(config_yaml)
    return config

def process_config(config_name):
    config = load_config(config_name)
    config.exp_dir = os.path.join(config.exp_base, "experiments", config.exp_name)
    
    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    if config.save_checkpoint_per_iter is not None:
        # create important directories for the experiment
        config.checkpoint_dir = os.path.join(config.exp_dir, "checkpoints/")
        config.out_dir = os.path.join(config.exp_dir, "out/")
        config.log_dir = os.path.join(config.exp_dir, "logs/")

        # will not create if already existing
        makedirs([config.checkpoint_dir, config.out_dir, config.log_dir])

        # Save the config to the exp_dir
        config_out = os.path.join(config.exp_dir, 'config.yaml')
        with open(config_out, 'w') as outfile:
            yaml.dump(config.toDict(), outfile)

    return config
    

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)