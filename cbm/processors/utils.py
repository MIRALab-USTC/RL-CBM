from numpy.core.fromnumeric import size
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from kornia import augmentation
import cbm.torch_modules.utils as ptu
import numpy as np

#use it before div255
class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4,aug=True):
        super().__init__()
        self.pad = pad
        self.aug = aug

    def forward(self, x):
        if self.aug:
            n, _, h, w = x.size()
            # assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            eps = 1.0 / (w + 2 * self.pad)
            arange = torch.linspace(-1.0 + eps,
                                    1.0 - eps,
                                    w + 2 * self.pad,
                                    device=x.device,
                                    dtype=x.dtype)[:w]
            eps_h = 1.0 / (h + 2 * self.pad)
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            arange_w = torch.linspace(-1.0 + eps_h,
                                    1.0 - eps_h,
                                    h + 2 * self.pad,
                                    device=x.device,
                                    dtype=x.dtype)[:h]
            arange_w = arange_w.unsqueeze(1).repeat(1, w).unsqueeze(2)
            # arange_w = arange_w.unsqueeze(0).repeat(w, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange_w], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

            shift = torch.randint(0,
                                2 * self.pad + 1,
                                size=(n, 1, 1, 2),
                                device=x.device,
                                dtype=x.dtype)
            shift[:,:,:,0] *= 2.0 / (w + 2 * self.pad)
            shift[:,:,:,1] *= 2.0 / (h + 2 * self.pad)

            grid = base_grid + shift
            return F.grid_sample(x,
                                grid,
                                padding_mode='zeros',
                                align_corners=False)
        else:
            return x


class RandomShiftsAug_old(nn.Module):
    def __init__(self, pad=4,aug=True):
        super().__init__()
        self.pad = pad
        self.aug = aug

    def forward(self, x):
        if self.aug:
            n, _, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            eps = 1.0 / (h + 2 * self.pad)
            arange = torch.linspace(-1.0 + eps,
                                    1.0 - eps,
                                    h + 2 * self.pad,
                                    device=x.device,
                                    dtype=x.dtype)[:h]
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

            shift = torch.randint(0,
                                2 * self.pad + 1,
                                size=(n, 1, 1, 2),
                                device=x.device,
                                dtype=x.dtype)
            shift *= 2.0 / (h + 2 * self.pad)

            grid = base_grid + shift
            return F.grid_sample(x,
                                grid,
                                padding_mode='zeros',
                                align_corners=False)
        else:
            return x
class CutoutColorAug(nn.Module):
    def __init__(self, size_min=15, size_max=30):
        super().__init__()
        self.size_min = size_min
        self.size_max = size_max

    def forward(self, x):
        aug_x = x.clone()
        n, c, h, w = x.size()
        assert h == w
        box_h = np.random.randint(self.size_min, self.size_max)
        box_w = np.random.randint(self.size_min, self.size_max)
        s_1, s_2 = np.random.randint(h), np.random.randint(w)
        e_1, e_2 = min(s_1+box_h, h), min(s_2+box_w, w)
        with torch.no_grad():
            color = ptu.randint(0, 256, size=(n, c, 1, 1))
            aug_x[:,:,s_1:e_1,s_2:e_2] = color
        return aug_x

class RandConvAug(nn.Module):
    def __init__(self, kernel_size=3, lam=0.7):
        super().__init__()
        assert (kernel_size-1)%2 == 0
        self.randconv = nn.Conv2d(3, 3, kernel_size, 1, (kernel_size-1)//2)
        self.randconv.to(ptu.device)
        self.lam = lam
        for param in self.randconv.parameters():
            param.requires_grad = False

    def reset_param(self):
        torch.nn.init.xavier_normal_(self.randconv.weight)
        if self.randconv.bias is not None:
            nn.init.constant_(self.randconv.bias, 0)
        
    def forward(self, x):
        self.reset_param()
        n, c, h, w = x.shape
        stack = c // 3
        x = x.reshape(n*stack, 3, h, w)
        noise = self.randconv(x)
        x = self.lam*x + (1-self.lam) * noise
        x = torch.clamp(x, 0, 255)
        return x.view(n,c,h,w)

class ImageAug(nn.Module):
    def __init__(self, image_aug=True):
        super().__init__()
        self.image_aug = image_aug
        self.trans_set = nn.ModuleList()
        self.names = []
        self.n_trans = 0

    def add_trans(self, trans, name=None):
        self.trans_set.append(trans)
        self.n_trans += 1
        if name is None:
            name = "aug%d"%self.n_trans
        self.names.append(name)
    
    def forward(self, x, return_name=False):
        n_trans = self.n_trans
        if (not self.image_aug) or (n_trans==0):
            name = None
        else:
            ind = np.random.randint(n_trans)
            trans = self.trans_set[ind]
            if len(x.shape) == 5:
                B,L,C,H,W = x.shape
                x = x.view(B,L*C,H,W)
                x = trans(x)
                x = x.view(B,L,C,H,W)
            else:
                x = trans(x)
            name = self.names[ind]
        self.aug_name = name
        if return_name:
            return x, name
        else:
            return x

if __name__ == "__main__":
    test = torch.rand(50,3,84,84)
    shift = RandomShiftsAug(pad=4)
    test_shift = shift(test)