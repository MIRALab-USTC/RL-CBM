from typing import Tuple
import dateutil.tz
import datetime
import time
import argparse
import shutil

from collections import OrderedDict

import json
import os
import os.path as osp
import random
import torch
import numpy as np

import cbm
from cbm.utils.logger import * 
from cbm.utils.initialize_utils import get_item, visit_all_items
import cbm.torch_modules.utils as ptu
import warnings
import copy


_mbrl_project_dir = osp.join(os.path.dirname(cbm.__file__), os.pardir)
_LOCAL_LOG_DIR = osp.join(_mbrl_project_dir, 'data')


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}

    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def replace_with_constant(json_dict):
    if 'constant' in json_dict:
        constant_dict = json_dict.pop('constant')
        _replace_with_constant(constant_dict, json_dict)

def _replace_with_constant(constant_dict, d):
    for k, v in d.items():
        if isinstance(v, str) and v[0] == '$':
            if v[1:] in constant_dict:
                d[k] = constant_dict[v[1:]]
        elif isinstance(v, dict) or isinstance(v, OrderedDict):
            _replace_with_constant(constant_dict, v)

def recursively_update_config(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) or isinstance(v, OrderedDict):
            if k in d1:
                assert isinstance(d1[k], dict) or isinstance(d1[k], OrderedDict)
                recursively_update_config(d1[k], d2[k])
            else:
                d1[k] = d2[k]
        else:
            d1[k] = v

def create_exp_name(
    exp_prefix, 
    seed=0
):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s-%s-%d" % (exp_prefix, timestamp, seed)


def create_log_dir(
    exp_prefix,
    seed=0,
    use_gpu=False,
    tag="",
    base_log_dir=None,
    files=None
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log directories be under this directory.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    cpu_or_gpu = "" if use_gpu else "-cpu"
    if len(tag) > 0:
        tag = '-' + tag 
    exp_prefix = exp_prefix + tag + cpu_or_gpu

    exp_name = create_exp_name(exp_prefix, seed)

    if base_log_dir is None:
        base_log_dir = _LOCAL_LOG_DIR

    if base_log_dir[0] == '~':
        base_log_dir = osp.expanduser(base_log_dir)
    log_dir = osp.join(base_log_dir, exp_prefix, exp_name)

    if osp.exists(log_dir):
        logger.log("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    
    # save critical code
    if files:
        code_dir = osp.join(log_dir,'code')
        os.makedirs(code_dir, exist_ok=True)
        for file_path in files:
            try:
                shutil.copy(file_path, osp.join(_mbrl_project_dir,code_dir))
            except:
                pass
    return log_dir


def setup_logger(
        exp_prefix="default",
        variant=None,
        log_level="WARNING",
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to

        base_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """

    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)
    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)

    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)

    logger.set_log_level(log_level)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir

def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time()*1e7)%4096
    np.random.seed(seed)    
    random.seed(seed)    
    torch.manual_seed(seed) #cpu    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    return seed

 
def try_eval(v):
    try:
        return eval(v)
    except:
        return v


def _set_config_by_k_v(config, k, v):
    v = try_eval(v)
    keys = k.split('.')
    find = False
    if len(keys) == 2: 
        if keys[0][:20] == "constant": # type of items
            config['constant'][keys[1]] = v
        elif keys[0][:6] == "type::": # type of items
            for _,item_type,_,kwargs in visit_all_items(config):
                if item_type == keys[0][5:]:
                    kwargs[keys[1]] = v
                    find = True
            if not find:
                warnings.warning("can not find type: %s"%keys[1])
        elif keys[0][:7] == "class::": # class of items
            for _,_,class_name,kwargs in visit_all_items(config):
                if class_name == keys[0][6:]:
                    kwargs[keys[1]] = v
                    find = True
            if not find:
                warnings.warning("can not find class: %s"%keys[1])
        else: # name of item
            for name,_,_,kwargs in visit_all_items(config):
                if name == keys[0]:
                    kwargs[keys[1]] = v
                    find = True
            if not find:
                warnings.warn("can not find item name: %s"%keys[1])
    elif len(keys) == 1:
        config['experiment'][keys[0]] = v
    else:
        raise NotImplementedError
    return config


def update_config(config, cmd_config):
    for k,v in cmd_config.items():
        _set_config_by_k_v(config, k, v)


def get_config_from_file(config_path):
    config = json.load(open(config_path, 'r'), object_pairs_hook=OrderedDict)

    # Command may be in the file 
    if 'cmd_config' in config:
        cmd_config = config.pop('cmd_config')
    else:
        cmd_config = {}

    # Config file can be based on a previous one
    if 'base_config_file' in config:
        base_config_file = config.pop('base_config_file')
        base_config = get_config_from_file(base_config_file)
        recursively_update_config(base_config, config)
        config = base_config

    update_config(config, cmd_config)

    return config


def run_single_experiment(config: OrderedDict):
    config = copy.deepcopy(config)
    experiment_kwargs = config['experiment']

    # Set seed. If not given, choose a seed randomly.
    seed = experiment_kwargs.get('seed', None)
    seed = set_global_seed(seed)
    experiment_kwargs['seed'] = seed
    logger.log("seed: %d"%seed)

    # Set device
    use_gpu = experiment_kwargs.get('use_gpu', False)
    ptu.set_gpu_mode(use_gpu)
    logger.log("device: %s"%ptu.device)

    # Set logger
    logger.reset()
    variant = copy.deepcopy(config)
    actual_log_dir = setup_logger(
        variant=variant,
        **experiment_kwargs
    )
    logger.log("log_dir: %s"%actual_log_dir)

    # Get algorithm
    config.pop('experiment')
    replace_with_constant(config)
    logger.log("Config:")
    logger.log(json.dumps(dict_to_safe_json(config), indent=2))
    algo = config.pop('algorithm')
    for k in algo:
        algo_class = algo[k]['class']
        algo_kwargs = algo[k]['kwargs']
        algo_kwargs['item_dict_config'] = config
        algo = get_item('algorithm', algo_class, algo_kwargs)
        logger.log("algorithm_name: %s"%k)
        
        algo.to(ptu.device)
        algo.train()

        
def _run_experiments(config_path: str, cmd_config: OrderedDict) -> None:
    # Read config from the file
    config = get_config_from_file(config_path)

    # Update config with command
    update_config(config, cmd_config)

    # Set default experimental prefix
    if 'exp_prefix' not in config['experiment']:
        exp_prefix = osp.basename(config_path)
        exp_prefix = exp_prefix.split('.')[0]
        config['experiment']['exp_prefix'] = exp_prefix

    # Run the experiment several times
    repeat = config['experiment'].pop('repeat', 1)
    for _ in range(repeat):
        run_single_experiment(config)


def run_experiments(config_path: str, cmd_config: OrderedDict) -> None:
    # Run experiments with several config files (in the given director)
    if osp.isdir(config_path):
        for file_name in os.listdir(config_path):
            if file_name[-5:] == '.json':
                json_path = osp.join(config_path, file_name)
                _run_experiments(json_path, cmd_config)
    # Run experiemtns with the given config file
    elif osp.isfile(config_path):
        _run_experiments(config_path, cmd_config)
    else:
        print("file does not exist!")


def parse_cmd() -> Tuple[str, OrderedDict]:
    p = argparse.ArgumentParser()
    p.add_argument('config_file', type=str)
    args, extras = p.parse_known_args()
    def foo(astr):
        if astr.startswith('--'):
            astr = astr[2:]
        else:
            raise RuntimeError('Keys must start with \"--\"')
        return astr
    cmd_config = [[foo(k),v] for k,v in zip(extras[::2],extras[1::2])]
    cmd_config = OrderedDict(cmd_config)
    return args.config_file, cmd_config