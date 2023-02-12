import numpy as np
import warnings
from collections import OrderedDict

from numpy.core.shape_base import block

from cbm.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from cbm.pools.trajectory_pool import TrajectoryPool
import cbm.torch_modules.utils as ptu
from cbm.utils.logger import logger
import random
import warnings
from termcolor import colored
import copy

class DomainPool(TrajectoryPool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        traj_len=None,  
        return_traj=None,
        compute_mean_std=False,
        ndomain_per_batch=2,
    ):
        super().__init__(env, max_size, traj_len, return_traj, compute_mean_std)
        self.ndomain_per_batch = ndomain_per_batch
        self._domain_index = {}
        self._domain_index_start = {}
        self._domain_index_stop = {}
        self._domain_size = {}
        self._domain2int = {}
        self._ndomain = 1
    
    def _get_default_fields(self):
        f_shape = self._frame_shape 
        a_shape = self._action_shape 
        field = {
            'frames': {
                'shape': f_shape,
                'type': np.uint8,
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
            'domain':{
                'shape': (1,),
                'type':np.uint8,
            }
        }
        if self._env_return_state:
            field['states'] = {
                'shape': self._state_shape,
                'type': np.float32
            }
            field['physics'] = {
                'shape':self._physics_shape,
                'type':np.float32
            }
        return field

    def update_index(self):
        self._path_len += 1
        # valid traj if its length is larger than _traj_len
        if self._path_len >= self.traj_len:
            self._index[self._index_stop] = self._stop 
            self._index_stop = (self._index_stop + 1) % self.max_size
            self._size += 1
            domain_index_stop = self._domain_index_stop[self._cur_domain]
            domain_index = self._domain_index[self._cur_domain]
            domain_index[domain_index_stop] = self._stop
            self._domain_index_stop[self._cur_domain] = (domain_index_stop+1)%self.max_size
            self._domain_size[self._cur_domain] += 1
            
        # drop invalid index
        # for example: _stop=0, traj_len=3, drop 2=0+3-1 
        first_index = self._index[self._index_start]
        if self._size>0 and first_index==(self._stop+self.traj_len-1) % self.max_size:
            self._index_start = (self._index_start + 1) % self.max_size
            self._size -= 1
            for domain, domain_index_start in self._domain_index_start.items():
                domain_first_index = self._domain_index[domain][domain_index_start]
                domain_size = self._domain_size[domain]
                if domain_size > 0 and first_index == domain_first_index:
                    self._domain_index_start[domain] = (domain_index_start+1)%self.max_size
                    self._domain_size[domain] -= 1

        # update _stop
        self._stop = (self._stop + 1) % self.max_size

    def set_domain(self, domain):
        self._cur_domain = domain
        if domain not in self._domain_index:
            self._domain2int[domain]=self._ndomain
            self._ndomain += 1
            self._domain_index[domain] = np.zeros((int(self.max_size),), dtype=int)
            self._domain_index_start[domain] = 0
            self._domain_index_stop[domain] = 0
            self._domain_size[domain] = 0

    def add_samples(self, samples):
        self.set_domain(samples['env_infos']['domain'])
        samples['domain']=np.array([self._domain2int[samples['env_infos']['domain']]])
        super().add_samples(samples)
        
    def get_batch_random_index(self, batch_size):
        domains = list(self._domain_index.keys())

        if self.ndomain_per_batch < 1:
            return TrajectoryPool.get_batch_random_index(self, batch_size)

        if min(list(self._domain_size.values())) == 0:
            for domain in domains:
                if self._domain_size[domain]==0:
                    domains.remove(domain)
        
        n = 0
        if len(domains) < self.ndomain_per_batch:
            for i in reversed(range(len(domains))):
                if batch_size%i == 0:
                    n=i
                    break
        else:
            n = self.ndomain_per_batch
            assert batch_size%n == 0
        domains = random.sample(domains, n)
        block_size = batch_size//n
        ind_list = []
        for domain in domains:
            ind_ind = np.random.randint(0, self._domain_size[domain], block_size)
            ind_ind = (ind_ind + self._domain_index_start[domain]) % self.max_size
            ind_list.append(self._domain_index[domain][ind_ind])
        ind = np.concatenate(ind_list)
        return ind
    
    def analyze_sample(self, batch_size, ndomain=0):
        ndomain_per_batch = self.ndomain_per_batch
        self.ndomain_per_batch = ndomain
        batch = self.random_batch_torch(batch_size)
        self.ndomain_per_batch = ndomain_per_batch
        return batch
    