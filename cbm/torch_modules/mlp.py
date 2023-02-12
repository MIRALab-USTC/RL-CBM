from typing import Iterable, List, Optional, Union, Tuple, Dict
import os.path as osp

import torch
import torch.nn as nn
from torch.nn.modules import module

import cbm.torch_modules.utils as ptu
from cbm.torch_modules.linear import get_fc
from cbm.utils.misc_untils import to_list

# A more simple implementation that directly uses nn.Linear. 
# However, it does not support EnsembleLinear.
def build_mlp(
    layer_size: List[int], 
    ensemble_size: Optional[int],
    activation: Union[str, List[str]] = "relu",
    output_activation: str = 'identity',
    **linear_kwargs: dict,
) -> Tuple[nn.Module, List[int]]:
    num_fc = len(layer_size) - 1
    act_name = to_list(activation, num_fc-1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]
    
    module_list = []
    final_layer_size = [layer_size[0]]  # for densenet
    in_features = layer_size[0]
    assert len(act_func) == num_fc
    for i in range(num_fc):
        fc = get_fc(in_features, layer_size[i+1], ensemble_size, i==num_fc, **linear_kwargs)
        module_list.append(fc)
        module_list.append(act_func[i])
        in_features = fc.final_out_features
        final_layer_size.append(in_features) # for densenet
    return nn.Sequential(*module_list), final_layer_size

# A more simple implementation that directly uses nn.Linear. 
# However, it does not support EnsembleLinear.
def build_mlp_v2(
    layer_size: List[int], 
    ensemble_size: Optional[int],
    activation: Union[str, List[str]] = "relu",
    output_activation: str = 'identity',
    **linear_kwargs: dict,
) -> Tuple[nn.Module, List[int]]:
    assert ensemble_size is None
    num_fc = len(layer_size) - 1
    act_name = to_list(activation, num_fc-1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]
    
    module_list = []
    assert len(act_func) == num_fc
    for i in range(num_fc):
        fc = nn.Linear(layer_size[i], layer_size[i+1], **linear_kwargs)
        module_list.append(fc)
        module_list.append(act_func[i])
    return nn.Sequential(*module_list), layer_size

class MLP(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_layers: List[int], 
        ensemble_size: Optional[int] = None,
        activation: Union[str, List[str]] = 'relu', 
        output_activation: str = 'identity',
        module_name: str = 'mlp',
        v1_or_v2: str = 'v1',
        **linear_kwargs
    ) -> None:
        #If ensemble is n
        #Given a tensor with shape (n,a,b) output (n,a,c)
        #Given a tensor with shape (a,b) output (n,a,c).  
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.module_name = module_name

        #get activation functions
        self.num_fc = len(hidden_layers) + 1
        layer_size = [input_size] + hidden_layers + [output_size]
        self.v1_or_v2 = v1_or_v2
        if v1_or_v2 == "v2":
            assert ensemble_size is None
        build_func = build_mlp if self.v1_or_v2 == "v1" else build_mlp_v2
        self.net, self.layer_size = build_func(
            layer_size, 
            ensemble_size, 
            activation, 
            output_activation,
            **linear_kwargs
        )
        self.min_output_dim = 2 if self.ensemble_size is None else 3

    def _forward_v1(self, x: torch.Tensor) -> torch.Tensor:
        if self.ensemble_size is None:
            max_output_dim  = x.dim()
        else:
            max_output_dim  = x.dim() + 1
        
        output = self.net(x)
        while output.dim() > max_output_dim:
            output = output.squeeze(0)
        return output

    def _forward_v2(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v1_or_v2 == "v1":
            return self._forward_v1(x)
        else:
            return self._forward_v2(x)

    def get_snapshot(self, key_must_have: str = '') -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        state_dict = self.state_dict()
        if key_must_have == '':
            new_state_dict = state_dict
        else:
            for k,v in state_dict.items():
                if key_must_have in k:
                    new_state_dict[k] = v
        return new_state_dict

    def load_snapshot(self, loaded_state_dict: dict, key_must_have: str = '') -> None:
        state_dict = self.state_dict()
        if key_must_have == '':
            state_dict = loaded_state_dict
        else:
            for k,v in loaded_state_dict.items():
                if key_must_have in k:
                    state_dict[k] = v
        self.load_state_dict(state_dict)

    def save(self, save_dir: str, net_id: Optional[int] = None) -> None:
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(save_dir, '%s.pt'%self.module_name)
        else:
            assert self.v1_or_v2 == "v1"
            net_name = 'net%d'%net_id
            file_path = osp.join(save_dir, '%s_%s.pt'%(self.module_name, net_name))
        state_dict = self.get_snapshot(net_name)
        torch.save(state_dict, file_path)
    
    def load(self, load_dir: str, net_id=None) -> None:
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        else:
            assert self.v1_or_v2 == "v1"
            net_name = 'net%d'%net_id
            file_path = osp.join(load_dir, '%s_%s.pt'%(self.module_name, net_name))
            if not osp.exists(file_path):
                file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        loaded_state_dict = torch.load(file_path)
        self.load_snapshot(loaded_state_dict, net_name)

    def get_weight_decay(self, weight_decays: Union[int, float, List[Union[int, float]]] = 0) -> torch.Tensor:
        assert self.v1_or_v2 == "v1"
        weight_decays = to_list(weight_decays, len(self.layer_size)-1)
        fcs = [fc for fc in self.net if hasattr(fc, "get_weight_decay")]
        assert len(fcs) == len(weight_decays)
        weight_decay_tensors = []
        for weight_decay, fc in zip(weight_decays, fcs):
            weight_decay_tensors.append(fc.get_weight_decay(weight_decay))
        return sum(weight_decay_tensors)


if __name__ == "__main__":
    import numpy as np
    x = torch.from_numpy(np.random.rand(2000,4))
    x = x.float()
    y = torch.from_numpy(np.random.rand(2000,1)) + (x*(x-1)).sum(-1)
    y = y.float()
    mlp1 = MLP(4, 1, [128,128], module_name='mlp1')

    mlp1_v2 = MLP(4, 1, [128,128], module_name='mlp2', v1_or_v2="v2")
    print(mlp1_v2)
    y1_v2 = mlp1_v2(x)
    mlp1_v2.save(osp.expanduser('~'))
    mlp2_v2 = MLP(4, 1, [128,128], module_name='mlp2', v1_or_v2="v2")
    mlp2_v2.load(osp.expanduser('~'))
    y2_v2 = mlp2_v2(x)
    print((y1_v2-y1_v2).abs().sum())


    mlp2 = MLP(4, 1, [128,128], 2, module_name='mlp2')
    pred1 = mlp1(x)
    pred2 = mlp2(x)
    mlp1.save(osp.expanduser('~'))
    mlp2.save(osp.expanduser('~'),0)
    mlp2.save(osp.expanduser('~'),1)
    opt1 = torch.optim.Adam(mlp1.parameters(), 0.01)
    opt2 = torch.optim.Adam(mlp2.parameters(), 0.01)
    for i in range(100):
        loss1 = ( (mlp1(x) - y) ** 2 ).mean()
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        loss2 = ( (mlp2(x) - y) ** 2 ).mean()
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
        print("%.6f\t\t%.6f"%(loss1.item(), loss2.item()))
    print(mlp1(x)-pred1)
    mlp1.load(osp.expanduser('~'))
    print(mlp1(x)-pred1)

    print(mlp2(x)-pred2)
    mlp2.load(osp.expanduser('~'),0)
    print(mlp2(x)-pred2)
    mlp2.load(osp.expanduser('~'),1)
    print(mlp2(x)-pred2)





