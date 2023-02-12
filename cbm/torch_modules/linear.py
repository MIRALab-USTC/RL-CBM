from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
import cbm.torch_modules.utils as ptu
import math

class MyLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        with_bias: bool = True,
        init_bias_constant: Union[int, float] = 0,
        connection: str = "simple",  #densenet, resnet, simple
        init_func_name: str = 'orthogonal_',
        init_kwargs: dict = {},
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if connection == "densenet":
            self.final_out_features = out_features + in_features
        else:
            self.final_out_features = out_features
            if connection == "resnet":
                assert out_features == in_features
            elif connection == "simple":
                pass
            else:
                raise NotImplementedError
        self.connection = connection

        self.init_func_name = init_func_name
        self.init_kwargs = init_kwargs

        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = with_bias
        self.init_bias_constant = init_bias_constant

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self._get_parameters()
        self.reset_parameters()
        
    def _get_parameters(self) -> None:
        self.weight, self.bias = self._creat_weight_and_bias()
    
    def _creat_weight_and_bias(self) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        weight = nn.Parameter(
            torch.empty(
                (self.in_features, self.out_features), 
                **self.factory_kwargs
            )
        )
        if self.with_bias:
            bias = nn.Parameter(
                torch.empty((1, self.out_features), **self.factory_kwargs)
            )
        else:
            bias = None
        return weight, bias

    def reset_parameters(self) -> None:
        self._reset_weight_and_bias(self.weight, self.bias)

    def _reset_weight_and_bias(
        self, 
        weight: nn.Parameter, 
        bias: Optional[nn.Parameter], 
        init_func_name: Optional[str] = None,
        init_kwargs: dict = {}
    ) -> None:
        if init_func_name is None:
            init_func_name = self.init_func_name
            init_kwargs = self.init_kwargs
        init_func = eval("nn.init."+init_func_name)
        init_func(weight.T, **init_kwargs)

        if bias is not None:
            if self.init_bias_constant is None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
            else:
                nn.init.constant_(bias, self.init_bias_constant)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 2:
            x = x.unsqueeze(0)
            
        if self.with_bias:
            output = x.matmul(self.weight) + self.bias
        else:
            output = x.matmul(self.weight)
        
        if self.connection == "densenet":
            output = torch.cat([output, x], -1)
        elif self.connection == "resnet":
            output = x + output
        
        return output

    def get_weight_decay(self, weight_decay: Union[int, float] = 0) -> torch.Tensor:
        return (self.weight ** 2).sum() * weight_decay * 0.5
        

class MyEnsembleLinear(MyLinear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        ensemble_size: int,
        **linear_kwargs
    ) -> None:
        self.ensemble_size = ensemble_size
        super().__init__(in_features, out_features, **linear_kwargs)
    
    def _get_parameters(self) -> None:
        self.weights, self.biases = [], []
        for i in range(self.ensemble_size):
            weight, bias = self._creat_weight_and_bias()
            weight_name, bias_name = 'weight_net%d'%i, 'bias_net%d'%i
            self.weights.append(weight)
            self.biases.append(bias)
            setattr(self, weight_name, weight)
            setattr(self, bias_name, bias)

    def reset_parameters(self) -> None:
        for w,s in zip(self.weights, self.biases):
            self._reset_weight_and_bias(w, s)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 3:
            x = x.unsqueeze(0)

        w = torch.stack(self.weights, 0)
        if self.with_bias:
            b = torch.stack(self.biases, 0)
            output = x.matmul(w) + b
        else:
            output = x.matmul(w)

        if self.connection == "densenet":
            target_shape = output.shape[:-1] + (self.out_features,)
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.expand(target_shape)
            output = torch.cat([output, x], -1)
        elif self.connection == "resnet":
            output = output + x
        
        return output

    def get_weight_decay(self, weight_decay: Union[int, float] = 0) -> torch.Tensor:
        decays = []
        for w in self.weights:
            decays.append((w ** 2).sum() * weight_decay * 0.5)
        return sum(decays)

def get_fc(
    in_features: int,
    out_features: int,
    ensemble_size: Optional[int],
    is_last: bool = False,
    **linear_kwargs
) -> Union[MyLinear, MyEnsembleLinear]:
    if is_last: # last layer should not be dense or res
        linear_kwargs['connection'] = "simple"        

    if ensemble_size is None:
        fc = MyLinear(
            in_features,
            out_features,
            **linear_kwargs
        )
    else:
        fc = MyEnsembleLinear(
            in_features, 
            out_features, 
            ensemble_size,
            **linear_kwargs
        )
    return fc
