import yaml
import os

def read_config_yaml():
    with open('configs.yml') as f:
        y = yaml.load(f)
        return y

def save_config_yaml(y):
    with open('configs.yml','w') as f:
        yaml.dump(y,f)

def get_all_filename(path):
    return os.listdir(path)

def check_path_or_creat(path):
    if not os.path.exists(path):
        os.makedirs(path)