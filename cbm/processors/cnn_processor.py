from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import cbm.torch_modules.utils as ptu
from cbm.processors.base_processor import Processor
from cbm.torch_modules.cnn import CNN, CNNTrans
from cbm.utils.misc_untils import to_list

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module, Processor):
    """Convolutional encoder for image-based observations."""
    def __init__(
        self, 
        env,
        output_size=50,
        **cnn_kwargs
    ):
        nn.Module.__init__(self)
        Processor.__init__(self)
        self.input_shape = env.observation_space.shape
        self.cnn_net = CNN(self.input_shape, output_size, **cnn_kwargs)
        self.output_shape = self.cnn_net.output_shape
        self.apply(weight_init)

    def process(self, obs):
        if len(obs.shape) == 5:
            return self.cnn_net.process_traj(obs)
        else:
            return self.cnn_net.process(obs)
    
    def feature_map(self, obs):
        return self.cnn_net.process_feature_map(obs)

    def forward(self, obs):
        return self.process(obs)

#note: normalize target
class CNNDecoder(nn.Module, Processor):
    """Convolutional encoder for image-based observations."""
    def __init__(
        self, 
        env,
        input_shape=288,
        **cnn_trans_kwargs
    ):
        nn.Module.__init__(self)
        Processor.__init__(self)
        self.output_shape = env.observation_space.shape
        self.cnn_trans_net = CNNTrans(input_shape, self.output_shape, **cnn_trans_kwargs)
        self.input_shape = self.cnn_trans_net.input_shape
        self.apply(weight_init)
    
    def process(self, latent):
        if len(latent.shape) == 3:
            return self.cnn_trans_net.process_traj(latent)
        else:
            return self.cnn_trans_net.process(latent)

    def forward(self, latent):
        return self.process(latent)

if __name__ == "__main__":
    net = CNNDecoder(input_size=288)
    torch_inpput = torch.zeros(1,288)
    print(net(torch_inpput))
