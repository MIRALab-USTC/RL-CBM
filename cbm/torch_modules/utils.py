import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math

"""
GPU wrappers
"""
_use_gpu = False
device = None
_gpu_id = 0
 
def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    if _use_gpu:
        set_device(gpu_id)

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, device=torch_device, **kwargs)
    else:
        return torch.FloatTensor(*args, **kwargs)

def LongTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    if _use_gpu:
        return torch.cuda.LongTensor(*args, device=torch_device, **kwargs)
    else:
        return torch.LongTensor(*args, **kwargs)


def from_numpy(data, *args, **kwargs):
    #return torch.from_numpy(*args, **kwargs).float().to(device)
    return torch.as_tensor(data, device=device).float()


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)

def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)

def eye(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.eye(*args, **kwargs, device=torch_device)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

def randn_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn_like(*args, **kwargs, device=torch_device)

def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs)

def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)

def rand_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand_like(*args, **kwargs, device=torch_device)

def randint(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randint(*args, **kwargs, device=torch_device)

def arange(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.arange(*args, **kwargs, device=torch_device)


############################ our utils ############################

class Swish(nn.Module):
    def __init__(self):        
        super().__init__()     
    def forward(self, x):        
        x = x * torch.sigmoid(x)        
        return x

class Identity(nn.Module):
    def __init__(self):        
        super().__init__()     
    def forward(self, x):            
        return x

def get_activation(act_name='relu'):
    activation_dict = {
        'relu': nn.ReLU(),
        'swish': Swish(),
        'tanh': nn.Tanh(),
        'identity': Identity(),
        'elu': nn.ELU()
    }
    if 'leaky_relu_' == act_name[:11]:
        return nn.LeakyReLU( eval(act_name[11:]) )
    else:
        return activation_dict[act_name]

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def np_to_pytorch_batch(np_batch):
    return {
        k: from_numpy(x)
        for (k, x) in np_batch.items()
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

def torch_to_np_info(torch_info):
    return {
        k: get_numpy(x)
        for (k, x) in torch_info.items()
    }


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class freeze_parameters_:
  def __init__(self, modules):
      """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = False

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]