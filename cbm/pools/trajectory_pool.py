import numpy as np
import warnings
from collections import OrderedDict

from cbm.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from cbm.pools.simple_pool import SimplePool
import cbm.torch_modules.utils as ptu
from cbm.utils.logger import logger
import copy

class TrajectoryPool(SimplePool):
    def __init__(
        self, 
        env, 
        max_size=1e6, 
        traj_len=None, 
        return_traj=None,
        compute_mean_std=False
    ):
        self._env_return_traj = env.return_traj
        if hasattr(env,'return_state'):
            self._env_return_state = env.return_state
            self._state_shape = env.state_shape
            # self._physics_shape = env.physics.get_state().shape
        self.frame_stack = env.frame_stack
        self._frame_shape = env.frame_shape #(3,84,84)
        self._channel = self._frame_shape[0]
        self._obs_before_reset = env.obs_before_reset
        super().__init__(env, max_size, compute_mean_std)

        if traj_len is None:
            self.traj_len = self.frame_stack + 1
        else:
            self.traj_len = traj_len

        # if return_traj=False: return batch for td-style update
        if return_traj is None:
            self.return_traj = env.return_traj 
        else:
            self.return_traj = return_traj


        assert self.traj_len > self.frame_stack
        assert max_size > 2*self.traj_len, "%d, %d"%(max_size, self.traj_len)

        self._index = np.zeros((int(self.max_size),), dtype=int)
        self._index_start = 0
        self._index_stop = 0
        self._size = 0
        self._path_len = 0


    def _get_default_fields(self):
        f_shape = self._frame_shape 
        a_shape = self._action_shape
        field =   {
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
        }
        if hasattr(self,'_env_return_state') and self._env_return_state:
            field['states'] = {
                'shape': self._state_shape,
                'type': np.float32
            }
            # field['physics'] = {
            #     'shape':self._physics_shape,
            #     'type':np.float32
            # }

        return field
    
    def get_last_frame(self, o):
        if self._env_return_traj:
            frame = o["frames"][-1]
        else:
            frame = o[:,-self._channel:]
        assert len(frame.shape) == 4
        return frame
    
    def update_index(self): 
        self._path_len += 1
        # valid traj if its length is larger than _traj_len
        # eg. traj_len, index =[5,6,7...,0,0,..]
        if self._path_len >= self.traj_len:
            self._index[self._index_stop] = self._stop 
            self._index_stop = (self._index_stop + 1) % self.max_size
            self._size += 1
            
        # drop invalid index
        # for example: _stop=0, traj_len=3, drop 2=0+3-1
        first_index = self._index[self._index_start]
        if self._size>0 and first_index==(self._stop+self.traj_len-1) % self.max_size:
            self._index_start = (self._index_start + 1) % self.max_size
            self._size -= 1

        # update _stop
        self._stop = (self._stop + 1) % self.max_size

    
    def _update_single_field(self, key, value=None):
        assert key in self.fields, (key, list(self.fields.keys()))
        # if not give (padding)
        if value is None:
            value = np.zeros(
                (1, *self.fields[key]['shape']), 
                dtype=self.fields[key]['type']
            )
        else:
            n_sample = len(value)
            assert n_sample==1 #TODO: support n!=1
        self.dataset[key][self._stop:self._stop+1] = value


    def _add_one_step_sample(self, sample):
        for k in self.fields:
            if k in sample:
                self._update_single_field(k, sample[k])
            else:
                self._update_single_field(k)
        self.update_index()

    
    def start_new_path(self, o):
        self._path_len = 0
        one_step_sample = {}
        if self._obs_before_reset == "repeat":
            one_step_sample['frames'] = self.get_last_frame(o)
        for i in range(self.frame_stack-1):
            self._add_one_step_sample(one_step_sample)


    # only support add_single_sample
    def add_samples(self, samples):
        one_step_sample = {}
        for k in self.fields:
            if k in samples:
                one_step_sample[k] = samples[k]
        o = samples["observations"]
        one_step_sample["frames"] = self.get_last_frame(o)
        if hasattr(self,'_env_return_state') and self._env_return_state:
            one_step_sample["states"] = samples['env_infos']['state']
            # one_step_sample["physics"] = np.array([samples['env_infos']['physics']])
        self._add_one_step_sample(one_step_sample)
        return 1


    def end_a_path(self, next_o):
        one_step_sample = {}
        one_step_sample['frames'] = self.get_last_frame(next_o) 
        self._add_one_step_sample(one_step_sample)


    def get_batch_random_index(self, batch_size):
        ind_ind = np.random.randint(0, self._size, batch_size)
        ind_ind = (ind_ind + self._index_start) % self.max_size
        ind = self._index[ind_ind]
        return ind

    def get_traj_index(self, index):
        traj_index = np.empty(
            shape=(len(index), self.traj_len),
            dtype=int
        )
        for i in range(self.traj_len):
            traj_index[:,-(i+1)] = (index-i)%self.max_size
        return traj_index

    def get_traj_by_index(self, traj_index, key):
        assert traj_index.shape[1] == self.traj_len
        flat_index = traj_index.flatten()
        data = self.dataset[key][flat_index]
        data = data.reshape(-1, self.traj_len, *self.fields[key]["shape"])
        return data

    def add_paths(self, paths):
        raise NotImplementedError

    def random_batch(self, batch_size, keys=None, without_keys=[]):
        raise NotImplementedError

    def random_batch_torch(self, batch_size, keys=None, without_keys=[]):
        index = self.get_batch_random_index(batch_size)
        traj_index = self.get_traj_index(index)
        if self.return_traj:
            torch_batch = {}
            for k in self.fields:
                data = self.get_traj_by_index(traj_index, k)            
                data = ptu.from_numpy(data)
                if k == "frames": #traj_len
                    torch_batch[k] = data  
                else: #traj_len -1
                    torch_batch[k] = data[:,:-1] # Note!
            return torch_batch
        else:
            torch_batch= {} 
            frame = self.get_traj_by_index(traj_index, 'frames')
            frame = ptu.from_numpy(frame)  # (B, traj_len, channel, w, h)
            frame = frame.view(-1, self.traj_len*self._channel, *self._frame_shape[1:])
            channel = self.frame_stack*self._channel
            torch_batch['observations'] = frame[:,:channel]
            torch_batch['next_observations'] = frame[:,-channel:]
            index = (index - 1) % self.max_size
            for k in ['actions', 'rewards', 'terminals']:
                torch_batch[k] = ptu.from_numpy(self.dataset[k][index])
            return torch_batch

    def shuffer_and_random_batch(self, batch_size, keys=None, without_keys=[]):
        raise NotImplementedError

    def get_data(self, keys=None, without_keys=[]):
        raise NotImplementedError

    def get_diagnostics(self):
        diagnostics =  OrderedDict([
            ('size', self._size),
        ])
        if logger.log_or_not(logger.INFO):
            pass
        return diagnostics

    def analyze_sample(self, batch_size):
        batch = self.random_batch_torch(batch_size)
        return batch
    