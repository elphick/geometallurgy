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


def get_column_config(config_dict: dict, var_map: dict, config_key: str = 'range') -> dict:
    res: dict = {}
    # populate from the config
    # var_map only includes mass-composition columns, no supplementary. vars are keys, cols are values
    composition_cols = [v for k, v in var_map.items() if k not in ['mass_wet', 'mass_dry', 'moisture']]

    for k, v in config_dict['vars'].items():
        if k == 'composition':
            for col in composition_cols:
                res[col] = v[config_key]
        elif k in list(var_map.keys()) and v.get(config_key):
            res[var_map[k]] = v[config_key]
    return res
