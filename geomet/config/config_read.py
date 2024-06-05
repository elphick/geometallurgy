import logging
from typing import Dict

import yaml


def read_yaml(file_path):
    with open(file_path, "r") as f:
        d_config: Dict = yaml.safe_load(f)
        if 'MC' != list(d_config.keys())[0]:
            msg: str = f'config file {file_path} is not a MassComposition config file - no MC key'
            logging.error(msg)
            raise KeyError(msg)
        return d_config['MC']


def read_flowsheet_yaml(file_path):
    with open(file_path, "r") as f:
        d_config: Dict = yaml.safe_load(f)
        if 'FLOWSHEET' != list(d_config.keys())[0]:
            msg: str = f'config file {file_path} is not a Flowsheet config file - no FLOWSHEET key'
            logging.error(msg)
            raise KeyError(msg)
        return d_config['FLOWSHEET']
