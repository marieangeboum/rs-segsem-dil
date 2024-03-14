import os
import yaml
import json
import random

def select_random_domains(domain_list, num_domains, seed=None):
    if seed is not None:
        random.seed(seed)
    
    selected_domains = random.sample(domain_list, num_domains)
    return selected_domains


def load_config_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_domain_sequence(config_file, new_domain_sequence):
    config = load_config_yaml(config_file)
    config['domain_sequence'] = new_domain_sequence
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


