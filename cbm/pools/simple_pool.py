from genericpath import isfile
import numpy as np
import warnings
from collections import OrderedDict

from cbm.utils.mean_std import RunningMeanStd
from cbm.pools.base_pool import Pool
from cbm.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from cbm.collectors.utils import path_to_samples
from cbm.utils.eval_util import create_stats_ordered_dict
from cbm.utils.logger import logger
import os.path as osp
import copy

class SimplePool(Pool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False):
        self.max_size = int(max_size)
        self.compute_mean_std = compute_mean_std
        self._observation_shape = env.observation_space.shape
        self._action_shape = env.action_space.shape
        self.fields = self._get_default_fields()
        self.sample_keys = list(self.fields.keys())
        self.dataset = {}
        if self.compute_mean_std:
            self.dataset_mean_std = {}
        for k, v in self.fields.items():
            self.initialize_field(k, v) 
        self._size = 0
        self._stop = 0

    def save(self, dir=None):
        if dir is None:
            dir = logger._snapshot_dir
        dataset_path = osp.join(dir, 'saved_dataset.npz')
        np.savez(dataset_path, **self.dataset)
        logger.log("save dataset to %s"%dataset_path)
        prop_dict = {k:v for k,v in self.__dict__.items() if not callable(v)}
        prop_dict["dataset"] = dataset_path
        instance_path = osp.join(dir, 'saved_pool.npy')
        np.save(instance_path, prop_dict, allow_pickle=True)
        logger.log("save pool to %s"%instance_path)
    
    def load(self, dir=None):
        if dir is None:
            dir = logger._snapshot_dir
        instance_path = osp.join(dir, 'saved_pool.npy')
        prop_dict = np.load(instance_path, allow_pickle=True).item()
        self.__dict__.update(prop_dict)
        logger.log("load pool from %s"%instance_path)
        dataset_path = prop_dict['dataset']
        if not osp.isfile(dataset_path):
            dataset_path = osp.join(dir, 'saved_dataset.npz')
        self.dataset = dict(np.load(dataset_path))
        logger.log("load dataset from %s"%dataset_path)
    
    def _get_default_fields(self):
        o_shape = self._observation_shape
        a_shape = self._action_shape
        return {
            'observations': {
                'shape': o_shape,
                'type': np.float32,
            },
            'next_observations': {
                'shape': o_shape,
                'type': np.float32,
            },
            'actions': {
                'shape': a_shape,
                'type': np.float32,
            },
            'rewards': {
                'shape': (1,),
                'type': np.float32,
            },
            'terminals': {
                'shape': (1,),
                'type': np.float32,
            },
        }

    def initialize_field(self, field_name, field_info):
        self.dataset[field_name] = np.empty(
            (int(self.max_size), *field_info['shape']), 
            dtype=field_info['type']
        )
        if self.compute_mean_std:
            self.dataset_mean_std[field_name] = RunningMeanStd(field_info['shape'])

    def get_size(self):
        return self._size

    def get_mean_std(self, keys=None, without_keys=[]):
        assert self.compute_mean_std
        keys = self._check_keys(keys, without_keys)
        mean_std = {
            k: [self.dataset_mean_std[k].mean,
                self.dataset_mean_std[k].std]
            for k in keys
        }
        return mean_std
    
    def random_batch(self, batch_size, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        return _random_batch_independently(self.dataset, batch_size, self._size, keys)

    def shuffer_and_random_batch(self, batch_size, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        for batch in _shuffer_and_random_batch(self.dataset, batch_size, self._size, keys):
            yield batch

    def get_data(self, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        data = {}
        for k in keys:
            temp_data = self.dataset[k]
            if self._size < self.max_size:
                data[k] = temp_data[:self._size]
            else:
                stop, size = self._stop, self._size
                data[k] = np.concatenate((temp_data[stop-size:], temp_data[:stop]))
        return data

    def _update_single_field(self, key, value):
        assert key in self.fields
        if self.compute_mean_std:
            self.dataset_mean_std[key].update(value)
        new_sample_size = len(value)
        max_size = self.max_size
        stop = self._stop
        new_stop = (stop + new_sample_size) % max_size
        if stop + new_sample_size >= max_size:
            self.dataset[key][stop:max_size] = value[:max_size-stop]
            self.dataset[key][:new_stop] = value[new_sample_size-new_stop:]
        else:
            self.dataset[key][stop:new_stop] = value
        return new_sample_size, new_stop

    def add_paths(self, paths):
        self.add_samples(path_to_samples(paths))
    
    def add_samples(self, samples):
        new_stop = None
        for k in self.fields:
            v = samples[k]
            if new_stop is None:
                new_sample_size, new_stop = self._update_single_field(k,v)
            else:
                _, _new_stop = self._update_single_field(k,v)
                assert _new_stop == new_stop
        self._stop = new_stop
        self._size = min(self.max_size, self._size + new_sample_size)
        return new_sample_size

    def _check_keys(self, keys, without_keys):
        if keys is None:
            keys = [k for k in self.fields.keys() if k not in without_keys]
        for k in keys:
            assert k in self.fields
        return keys
        
    def get_diagnostics(self):
        diagnostics =  OrderedDict([
            ('size', self._size),
        ])
        if logger.log_or_not(logger.INFO):
            data = self.get_data(['rewards', 'terminals'])
            diagnostics.update(create_stats_ordered_dict(
                'Reward',
                data['rewards'],
            ))
            d = data['terminals']
            diagnostics['Done Ratio'] = d.sum() / self._size
        return diagnostics


        