import os
from copy import deepcopy
import json
import torch
import numpy as np
import random

from utils.spec_reader import SpecReader

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def settings_to_dict(a_settings):
    new_dict = {}
    for k, v in a_settings.items():
        new_dict[k] = v.value
    return new_dict

def dict_to_settings(s_dict):
    return SpecReader(None, s_dict)

def clean_dict_for_json(a_dict):
    if is_jsonable(a_dict):
        return deepcopy(a_dict)
    else:
        if isinstance(a_dict, dict):
            new_dict = dict()
            for k in a_dict.keys():
                new_dict[k] = clean_dict_for_json(a_dict[k])
        else:
            return None
        return new_dict


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


def find_save_path(dir, trial_id):
    """
    Create a directory to save notebooks and arguments. Adds 100 to the trial id if a directory already exists.

    Params
    ------
    - dir (str)
        Main saving directory
    - trial_id (int)
        Trial identifier
    """
    i = 0
    while True:
        save_dir = dir + '/' + str(trial_id + i * 100) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i += 1
    return save_dir

def save_path_use_prefix(prefix, dir, trial_id):
    return os.path.join(dir, prefix, f'config{trial_id}')